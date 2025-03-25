"""Module for fine-tuning."""

# standard library imports
import argparse
import os
import time

# related third party imports
import structlog
from yacs.config import CfgNode
from langfuse.decorators import langfuse_context, observe

# local application/library specific imports
from data_loader.build import build_dataset
from tools.configurator import check_cfg, load_configs, save_config, convert_to_dict
from tools.utils import (
    delete_previous_content,
    print_elapsed_time,
    write_pickle,
    set_seed,
)
from prompt.few_shot_prompt import df_to_listdict
from tools.constants import TRAIN, VALIDATION, TEST
from prompt.build import build_prompt
from prompt.json_schema import MCQAnswer
from model.build import build_model
from tools.evaluate import evaluate, predict
from example_formatter.build import build_example_formatter


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
        session_id=f"{cfg.ID}~Run{run_n}",
        # user_id="session-1234",
        metadata=convert_to_dict(cfg),
        tags=["dry-run" if args.dry_run else "full-run"],
    )
    # get the langchain handler for the current trace
    langfuse_handler = langfuse_context.get_current_langchain_handler()

    start_time = time.time()
    print("\n", "*" * 10, f"Run: {run_n}/{cfg.RUNS}", "*" * 10)

    # load data
    datasets = build_dataset(cfg.LOADER, cfg.SEED + run_n)
    # subset
    if args.dry_run:
        logger.info("Dry run: using only 10 observations")
        datasets[VALIDATION] = datasets[VALIDATION].iloc[:10, :]
        datasets[TEST] = datasets[TEST].iloc[:10, :]

    # dataframes
    datasets_fmt = build_example_formatter(
        example_formatter_cfg=cfg.EXAMPLE_FORMATTER,
        datasets=datasets,
    )

    # list of dicts
    list_train = df_to_listdict(datasets_fmt[TRAIN])
    list_val = df_to_listdict(datasets_fmt[VALIDATION])
    list_test = df_to_listdict(datasets_fmt[TEST])  # noqa

    # seed
    set_seed(cfg.SEED + run_n)

    # prompt
    prompt, _ = build_prompt(cfg=cfg, examples=list_train)

    # model
    model = build_model(model_cfg=cfg.MODEL)
    if cfg.MODEL.STRUCTURED_OUTPUT:
        model = model.with_structured_output(
            MCQAnswer, include_raw=True
        )  # TODO: make flexible

    # chain
    chain = prompt | model

    # predict
    val_preds = predict(
        chain=chain,
        data=list_val,
        prefix="val",
        structured=cfg.MODEL.STRUCTURED_OUTPUT,
        json_schema=MCQAnswer,
        langfuse_handler=langfuse_handler,
    )
    # TODO: also for test set

    # evaluate
    val_result = evaluate(
        preds_validated=val_preds["val_preds_validated"],
        dataset=datasets[VALIDATION],
        prefix="val",
    )
    # test_result = evaluate(preds_validated=preds_validated, dataset=datasets[TEST], prefix="test")

    write_pickle(
        {
            **val_preds,
            **val_result,
            # **test_result,
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
    configs = load_configs(args.config)

    # remove previous contents (take dir form first cfg)
    delete_previous_content(configs[0])

    # logical checks before start running
    for cfg in configs:
        check_cfg(cfg)

    for cfg in configs:
        print("\n", "=" * 10, f"Config: {cfg.ID}", "=" * 10)

        # start experiment loop
        for run_n in range(1, cfg.RUNS + 1):
            run_single_cfg(cfg=cfg, run_n=run_n, args=args)

        save_config(cfg, save_dir=cfg.OUTPUT_DIR, fname=cfg.ID)


if __name__ == "__main__":
    main()
