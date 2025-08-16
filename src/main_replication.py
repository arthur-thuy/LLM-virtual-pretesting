"""Script for replicating student behaviour."""

# standard library imports
import argparse
import os
import time

# NOTE: load environment variables
from tools.utils import load_env  # isort:skip

load_env(os.path.join("..", ".env"))  # noqa

# related third party imports
import structlog
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from yacs.config import CfgNode

# local application/library specific imports
from data_loader.build import build_replicate_dataset
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
from tools.irt_estimator import apply_student_scale_map
from tools.constants import TEST, TRAIN, VALIDATION, VALLARGE, VALSMALL  # noqa
from tools.data_manager.utils import bring_correct_option_forward
from tools.evaluate import evaluate_replication, predict
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
def run_single_cfg(cfg: CfgNode, run_n: int, args, langfuse_session: Langfuse) -> None:
    """Run a single configuration."""
    # update trace attributes (e.g, name, session_id, user_id)
    langfuse_context.update_current_trace(
        # name="custom-trace",
        session_id=f"{cfg.ID}~Run{run_n}",
        metadata=convert_to_dict(cfg),
        tags=["dry-run" if args.dry_run else "full-run"],
    )
    # get the langchain handler for the current trace
    langfuse_handler = langfuse_context.get_current_langchain_handler()
    trace_id = langfuse_context.get_current_trace_id()

    start_time = time.time()
    print("\n", "*" * 10, f"Run: {run_n}/{cfg.RUNS}", "*" * 10)

    # load data
    datasets = build_replicate_dataset(cfg.LOADER)
    # choose small or large validation set
    if cfg.LOADER.RUN_LARGE_VAL:
        datasets[VALIDATION] = datasets.pop(VALLARGE)
        datasets.pop(VALSMALL)
    else:
        datasets[VALIDATION] = datasets.pop(VALSMALL)
        datasets.pop(VALLARGE)
    logger.info(
        "Choosing validation set",
        name=(VALLARGE if cfg.LOADER.RUN_LARGE_VAL else VALSMALL),
        num_interactions=len(datasets[VALIDATION]),
    )

    if cfg.CONTEXT_TYPE == "misconceptions":  # NOTE: alternative is "snippets"
        # bring correct option to first place
        logger.info("Misconceptions as context: bringing correct option forward")
        datasets[TRAIN] = datasets[TRAIN].apply(
            bring_correct_option_forward,
            is_interaction=True,
            axis=1,
        )

    # subset
    if args.dry_run:
        logger.info("Dry run: using only 10 observations")
        datasets[VALIDATION] = datasets[VALIDATION].loc[list(range(0, 100, 10))]

    # dataframes
    datasets_fmt = build_example_formatter(
        example_formatter_cfg=cfg.EXAMPLE_FORMATTER.INTERACTIONS,
        datasets=datasets,
        is_interaction=True,
    )

    # get student scale mapping
    student_scale_map, student_scale_str = build_student_scale(cfg=cfg)
    # apply student scale map to predefined groups!
    datasets_fmt = apply_student_scale_map(
        interactions=datasets_fmt, student_scale_map=student_scale_map
    )

    # list of dicts
    list_train = df_to_listdict(datasets_fmt[TRAIN])
    list_val = df_to_listdict(datasets_fmt[VALIDATION])
    # list_test = df_to_listdict(datasets_fmt[TEST])  # noqa  # TODO

    # seed
    set_seed(cfg.SEED + run_n)

    # structured output
    StrucOutput = build_structured_outputter(cfg.STRUCTURED_OUTPUTTER)

    # prompt
    prompt, _ = build_prompt(
        cfg=cfg,
        examples=list_train,
        struc_output=StrucOutput,
        student_scale_str=student_scale_str,
        q_ids_train=None,
    )

    # model
    model = build_model(model_cfg=cfg.MODEL)
    if cfg.MODEL.NATIVE_STRUCTURED_OUTPUT:
        model = model.with_structured_output(StrucOutput, include_raw=True)

    # chain
    chain = prompt | model

    # predict & evaluate
    val_preds_raw = predict(
        chain=chain,
        data=list_val,
        prefix="val",
        structured=cfg.MODEL.NATIVE_STRUCTURED_OUTPUT,
        json_schema=StrucOutput,
        langfuse_handler=langfuse_handler,
    )
    val_metrics, val_preds = evaluate_replication(
        preds_validated=val_preds_raw["val_preds_validated"],
        dataset=datasets[VALIDATION],  # unformatted dataset!
        prefix="val",
        langfuse_session=langfuse_session,
        trace_id=trace_id,
    )

    # TODO: also for test set
    # test_metrics, test_preds = evaluate(
    #     preds_validated=test_preds["test_preds_validated"],
    #     dataset=datasets[TEST],
    #     prefix="test",
    #     langfuse_session=langfuse_session,
    #     trace_id=trace_id,
    # )

    write_pickle(
        {
            "metrics": {**val_metrics},
            "preds_raw": {**val_preds_raw},
            "preds": {**val_preds},
            "val_data": datasets_fmt[VALIDATION],
        },
        save_dir=os.path.join(cfg.OUTPUT_DIR, cfg.ID),
        fname=f"run_{run_n}",
    )
    print_elapsed_time(start_time, run_n)
    langfuse_handler.flush()


def main() -> None:
    """Run experiment."""
    args = parser.parse_args()

    # config
    configs = load_configs(args.config, problem_type="replicate")

    # remove previous contents (take dir form first cfg)
    delete_previous_content(configs[0].OUTPUT_DIR)

    # logical checks before start running
    for cfg in configs:
        check_cfg(cfg, problem_type="replicate")

    # langfuse
    langfuse_session = Langfuse()

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
            print("\n", "=" * 10, f"Config: {cfg.ID}", "=" * 10)
            print(f"Config {i + 1}/{len(configs)}")

            # start experiment loop
            for run_n in range(1, cfg.RUNS + 1):
                run_single_cfg(
                    cfg=cfg,
                    run_n=run_n,
                    args=args,
                    langfuse_session=langfuse_session,
                )
            save_config(cfg, save_dir=cfg.OUTPUT_DIR, fname=cfg.ID)
            # try:
            #     for run_n in range(1, cfg.RUNS + 1):
            #         run_single_cfg(
            #             cfg=cfg,
            #             run_n=run_n,
            #             args=args,
            #             langfuse_session=langfuse_session,
            #         )
            #     save_config(cfg, save_dir=cfg.OUTPUT_DIR, fname=cfg.ID)
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
