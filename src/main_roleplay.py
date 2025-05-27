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
from yacs.config import CfgNode
from langfuse.decorators import langfuse_context, observe
from langfuse import Langfuse

# local application/library specific imports
from data_loader.build import build_roleplay_dataset
from tools.configurator import check_cfg, load_configs, save_config, convert_to_dict
from tools.utils import (
    delete_previous_content,
    print_elapsed_time,
    write_pickle,
    set_seed,
)
from prompt.utils import df_to_listdict
from tools.constants import (
    TRAIN,
    TEST,
    VALIDATION,
    QUESTION_ID,
)  # noqa
from tools.irt_estimator import group_student_levels, explode_student_levels
from prompt.build import build_prompt
from model.build import build_model
from tools.evaluate import evaluate_roleplay, predict
from example_formatter.build import build_example_formatter
from structured_outputter.build import build_structured_outputter


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
def run_single_cfg(cfg: CfgNode, run_n: int, args, langfuse_session: Langfuse) -> None:
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
    trace_id = langfuse_context.get_current_trace_id()

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

    # dataframes
    interact_train_fmt = build_example_formatter(
        example_formatter_cfg=cfg.EXAMPLE_FORMATTER,
        datasets={TRAIN: interact_train},
        is_interaction=True,
    )[TRAIN]
    questions_fmt = build_example_formatter(
        example_formatter_cfg=cfg.EXAMPLE_FORMATTER,
        datasets=questions,
        is_interaction=False,
    )

    # Compute IRT parameters and group students
    interact_train_fmt = group_student_levels(
        df_interactions=interact_train_fmt, num_groups=cfg.ROLEPLAY.NUM_STUDENT_LEVELS
    )
    print(interact_train_fmt.columns)

    # create one row for each student level
    for split, df_q in questions_fmt.items():
        questions_fmt[split] = explode_student_levels(
            df_questions=df_q, num_groups=cfg.ROLEPLAY.NUM_STUDENT_LEVELS
        )

    # list of dicts
    list_train = df_to_listdict(interact_train_fmt)
    list_val = df_to_listdict(questions_fmt[VALIDATION])
    # list_test = df_to_listdict(questions_fmt[TEST])  # TODO

    # seed
    set_seed(cfg.SEED + run_n)

    # structured output
    StrucOutput = build_structured_outputter(cfg.STRUCTURED_OUTPUTTER)

    # prompt
    q_ids_train = list(questions_fmt[TRAIN][QUESTION_ID].unique())
    prompt, _ = build_prompt(
        cfg=cfg, examples=list_train, q_ids_train=q_ids_train, struc_output=StrucOutput
    )

    # model
    model = build_model(model_cfg=cfg.MODEL)
    if cfg.MODEL.NATIVE_STRUCTURED_OUTPUT:
        model = model.with_structured_output(StrucOutput, include_raw=True)

    # chain
    chain = prompt | model

    # print(f"{questions_fmt[VALIDATION].columns=}")  # TODO: remove

    # predict & evaluate
    # TODO: need to save the student level of each question in the validation set
    val_preds_raw = predict(
        chain=chain,
        data=list_val,
        prefix="val",
        structured=cfg.MODEL.NATIVE_STRUCTURED_OUTPUT,
        json_schema=StrucOutput,
        langfuse_handler=langfuse_handler,
    )
    val_metrics, val_preds = evaluate_roleplay(
        preds_validated=val_preds_raw["val_preds_validated"],
        dataset=questions_fmt[VALIDATION],
        prefix="val",
    )

    # TODO: compute IRT on all output preds
    # TODO: evaluate predicted difficulty on questions to true difficulty

    # compute_q_difficulty()

    write_pickle(
        {
            "metrics": {**val_metrics},
            "preds_raw": {**val_preds_raw},
            "preds": {**val_preds},
        },
        save_dir=os.path.join(cfg.OUTPUT_DIR_ROLEPLAY, cfg.ID),
        fname=f"run_{run_n}",
    )
    print_elapsed_time(start_time, run_n)
    langfuse_handler.flush()


def main() -> None:
    """Run experiment."""
    args = parser.parse_args()

    # config
    configs = load_configs(args.config, freeze=False)

    # remove previous contents (take dir form first cfg)
    delete_previous_content(configs[0].OUTPUT_DIR_ROLEPLAY)

    # logical checks before start running
    for cfg in configs:
        check_cfg(cfg)

    # langfuse
    langfuse_session = Langfuse()

    for cfg in configs:
        print("\n", "=" * 10, f"Config: {cfg.ID}", "=" * 10)

        # start experiment loop
        for run_n in range(1, cfg.RUNS + 1):
            run_single_cfg(
                cfg=cfg, run_n=run_n, args=args, langfuse_session=langfuse_session
            )

        save_config(cfg, save_dir=cfg.OUTPUT_DIR_ROLEPLAY, fname=cfg.ID)


if __name__ == "__main__":
    main()
