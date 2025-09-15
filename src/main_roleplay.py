"""Script for LLM roleplaying."""

# standard library imports
import argparse
import os
import time

# NOTE: load environment variables
from tools.utils import load_env  # isort:skip

load_env(os.path.join("..", ".env"))  # noqa

# related third party imports
import structlog
from langfuse.decorators import langfuse_context, observe
from yacs.config import CfgNode

# local application/library specific imports
from data_loader.build import build_roleplay_dataset
from example_formatter.build import build_example_formatter
from model.build import build_model
from prompt.build import build_prompt
from prompt.utils import df_to_listdict
from structured_outputter.build import build_structured_outputter
from student_scale.build import build_student_scale
from tools.configurator import (
    check_cfg,
    convert_to_dict,
    get_configs_out,
    load_configs,
    save_config,
    check_config_equivalence,
)
from tools.constants import (  # noqa
    QUESTION_ID,
    TEST,
    TRAIN,
    VALIDATION,
)
from tools.data_manager.utils import bring_correct_option_forward
from tools.evaluate import evaluate_q_difficulty, evaluate_roleplay, predict
from tools.irt_estimator import (
    explode_student_levels,
    apply_student_scale_map,
)
from tools.utils import (
    delete_previous_content,
    print_elapsed_time,
    set_seed,
    write_pickle,
)

# set up logger
logger = structlog.get_logger(__name__)

parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument(
    "config",
    type=str,
    help="config file path",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    default=False,
    help="predict small number of examples",
)


# Create a trace via Langfuse decorators and get a Langchain Callback handler for it
@observe()  # automtically log function as a trace to Langfuse
def run_single_cfg(cfg: CfgNode, run_n: int, args) -> None:
    """Run a single configuration."""
    # update trace attributes (e.g, name, session_id, user_id)
    langfuse_context.update_current_trace(
        # name="custom-trace",
        session_id=f"{cfg.ID_ROLEPLAY}~Run{run_n}",
        metadata=convert_to_dict(cfg),
        tags=["roleplay", "dry-run" if args.dry_run else "full-run"],
    )
    # get the langchain handler for the current trace
    langfuse_handler = langfuse_context.get_current_langchain_handler()

    start_time = time.time()
    print("\n", "*" * 10, f"Run: {run_n}/{cfg.RUNS}", "*" * 10)

    # build dataset
    questions, interact_train = build_roleplay_dataset(
        loader_cfg=cfg.LOADER,
    )

    # subset for dry-run
    if args.dry_run:
        logger.info("Dry run: using only 2 questions")
        questions[VALIDATION] = questions[VALIDATION].iloc[:2, :]
        questions[TEST] = questions[TEST].iloc[:2, :]

    if cfg.CONTEXT_TYPE == "misconceptions":  # NOTE: alternative is "snippets"
        # bring correct option to first place
        logger.info("Misconceptions as context: bringing correct option forward")
        interact_train = interact_train.apply(
            bring_correct_option_forward,
            is_interaction=True,
            axis=1,
        )

    # get student scale mapping
    student_scale_map, student_scale_str = build_student_scale(cfg=cfg)
    # create one row for each student level
    for split, df_q in questions.items():
        questions[split] = explode_student_levels(
            df_questions=df_q, student_scale_map=student_scale_map
        )

    # format questions/interactions for prompt
    interact_train_fmt = build_example_formatter(
        example_formatter_cfg=cfg.EXAMPLE_FORMATTER.INTERACTIONS,
        datasets={TRAIN: interact_train},
        is_interaction=True,
    )[TRAIN]
    questions_fmt = build_example_formatter(
        example_formatter_cfg=cfg.EXAMPLE_FORMATTER.QUESTIONS,
        datasets=questions,
        is_interaction=False,
    )

    # apply student scale map to predefined groups!
    interact_train_fmt = apply_student_scale_map(
        interactions={TRAIN: interact_train_fmt}, student_scale_map=student_scale_map
    )[TRAIN]
    questions_fmt = apply_student_scale_map(
        interactions=questions_fmt, student_scale_map=student_scale_map
    )

    # list of dicts
    list_train = df_to_listdict(interact_train_fmt)
    list_val = df_to_listdict(questions_fmt[VALIDATION])
    list_test = df_to_listdict(questions_fmt[TEST])  # TODO

    # seed
    set_seed(cfg.SEED + run_n)

    # structured output
    StrucOutput = build_structured_outputter(cfg.STRUCTURED_OUTPUTTER)

    # prompt
    if cfg.LOADER.JOIN_KEY is None:
        q_ids_train = None
    else:
        q_ids_train = list(questions_fmt[TRAIN][QUESTION_ID].unique().tolist())
    prompt, _ = build_prompt(
        cfg=cfg,
        examples=list_train,
        struc_output=StrucOutput,
        student_scale_str=student_scale_str,
        q_ids_train=q_ids_train,
    )

    # model
    model = build_model(model_cfg=cfg.MODEL)
    if cfg.MODEL.NATIVE_STRUCTURED_OUTPUT:
        model = model.with_structured_output(StrucOutput, include_raw=True)

    # chain
    chain = prompt | model

    # compute answer correctness per student level in train set
    if cfg.LOADER.JOIN_KEY is None:
        train_student_group_correctness = None
    else:
        train_student_group_correctness = (
            interact_train.groupby("student_level_group")["student_option_correct"]
            .mean()
            .to_numpy()
        )

    # predict & evaluate
    if cfg.LOADER.RUN_VAL:
        val_preds_raw = predict(
            chain=chain,
            data=list_val,
            prefix="val",
            structured=cfg.MODEL.NATIVE_STRUCTURED_OUTPUT,
            json_schema=StrucOutput,
            langfuse_handler=langfuse_handler,
        )
        val_metrics_answers, val_preds_answers = evaluate_roleplay(
            preds_validated=val_preds_raw["val_preds_validated"],
            dataset=questions[VALIDATION],  # unformatted dataset!
            prefix="val",
            student_group_correctness=train_student_group_correctness,
            only_kt=("kt" in cfg.STRUCTURED_OUTPUTTER.NAME),
        )
        # compute IRT and evaluate question difficulty
        val_metrics_qdiff, val_preds_qdiff = evaluate_q_difficulty(
            preds_validated=val_preds_raw["val_preds_validated"],
            dataset=questions[VALIDATION],
            prefix="val",
            difficulty_range=cfg.LOADER.DIFFICULTY_RANGE,
            only_kt=("kt" in cfg.STRUCTURED_OUTPUTTER.NAME),
        )
    else:
        val_preds_raw = {}
        val_metrics_answers, val_preds_answers = {}, {}
        val_metrics_qdiff, val_preds_qdiff = {}, {}

    if cfg.LOADER.RUN_TEST:
        test_preds_raw = predict(
            chain=chain,
            data=list_test,
            prefix="test",
            structured=cfg.MODEL.NATIVE_STRUCTURED_OUTPUT,
            json_schema=StrucOutput,
            langfuse_handler=langfuse_handler,
        )
        test_metrics_answers, test_preds_answers = evaluate_roleplay(
            preds_validated=test_preds_raw["test_preds_validated"],
            dataset=questions[TEST],  # unformatted dataset!
            prefix="test",
            student_group_correctness=train_student_group_correctness,
            only_kt=("kt" in cfg.STRUCTURED_OUTPUTTER.NAME),
        )
        # compute IRT and evaluate question difficulty
        test_metrics_qdiff, test_preds_qdiff = evaluate_q_difficulty(
            preds_validated=test_preds_raw["test_preds_validated"],
            dataset=questions[TEST],
            prefix="test",
            difficulty_range=cfg.LOADER.DIFFICULTY_RANGE,
            only_kt=("kt" in cfg.STRUCTURED_OUTPUTTER.NAME),
        )
    else:
        test_preds_raw = {}
        test_metrics_answers, test_preds_answers = {}, {}
        test_metrics_qdiff, test_preds_qdiff = {}, {}

    log_data = {
        "metrics": {
            **val_metrics_qdiff,
            **val_metrics_answers,
            **test_metrics_qdiff,
            **test_metrics_answers,
        },
        # "preds_raw": {**val_preds_raw, **test_preds_raw},
        "preds_answers": {**val_preds_answers, **test_preds_answers},
        "preds_qdiff": {**val_preds_qdiff, **test_preds_qdiff},
        "student_group_correctness": train_student_group_correctness,
        "student_scale_map": student_scale_map,
    }
    if cfg.LOADER.RUN_VAL:
        log_data["val_data"] = questions_fmt[VALIDATION]
    if cfg.LOADER.RUN_TEST:
        log_data["test_data"] = questions_fmt[TEST]

    write_pickle(
        log_data,
        save_dir=os.path.join(cfg.OUTPUT_DIR, cfg.ID_ROLEPLAY),
        fname=f"run_{run_n}",
    )
    print_elapsed_time(start_time, run_n)
    langfuse_handler.flush()


def main() -> None:
    """Run experiment."""
    args = parser.parse_args()

    # config
    configs = load_configs(args.config, problem_type="roleplay", freeze=False)

    # remove previous contents (take dir form first cfg)
    delete_previous_content(configs[0].OUTPUT_DIR)

    # logical checks before start running
    for cfg in configs:
        check_cfg(cfg, problem_type="roleplay")

    previous_experiment_names = []
    previous_configs = []
    for EXP_NAME in previous_experiment_names:
        print(EXP_NAME)
        previous_configs.extend(get_configs_out(EXP_NAME))
    errors = []
    for i, cfg in enumerate(configs):
        already_evaluated = False
        for prev_cfg in previous_configs:
            if check_config_equivalence(prev_cfg, cfg):
                already_evaluated = True
                break
        if not already_evaluated:
            print("\n", "=" * 10, f"Config: {cfg.ID_ROLEPLAY}", "=" * 10)
            print(f"Config {i + 1}/{len(configs)}")

            # start experiment loop
            for run_n in range(1, cfg.RUNS + 1):
                run_single_cfg(
                    cfg=cfg,
                    run_n=run_n,
                    args=args,
                )
            save_config(cfg, save_dir=cfg.OUTPUT_DIR, fname=cfg.ID_ROLEPLAY)
            # try:
            #     for run_n in range(1, cfg.RUNS + 1):
            #         run_single_cfg(
            #             cfg=cfg,
            #             run_n=run_n,
            #             args=args,
            #             langfuse_session=langfuse_session,
            #         )
            #     save_config(cfg, save_dir=cfg.OUTPUT_DIR, fname=cfg.ID_ROLEPLAY)
            # except Exception as e:
            #     errors.append((cfg.ID, e))
            #     logger.error(
            #         "Error occurred during the experiment",
            #         config=cfg.ID,
            #         error=str(e),
            #     )
        else:
            print(cfg.ID, "already evaluated.")

    if len(errors) > 0:
        logger.error(
            "Errors occurred during the following experiments",
            configs=errors,
        )
    else:
        logger.info("All experiments completed successfully.")


if __name__ == "__main__":
    main()
