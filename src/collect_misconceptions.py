"""Script for collecting misconceptions of train questions."""

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
from data_loader.build import build_roleplay_dataset
from example_formatter.build import build_example_formatter
from model.build import build_model
from prompt.build import build_prompt
from prompt.utils import df_to_listdict
from structured_outputter.build import build_structured_outputter
from tools.configurator import (
    check_cfg,
    convert_to_dict,
    load_configs,
    save_config,
)
from tools.constants import TEST, TRAIN, VALIDATION  # noqa
from tools.evaluate import predict
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
        session_id=f"{cfg.ID_MISCON}~Run{run_n}",
        metadata=convert_to_dict(cfg),
        tags=["dry-run" if args.dry_run else "full-run"],
    )
    # get the langchain handler for the current trace
    langfuse_handler = langfuse_context.get_current_langchain_handler()

    start_time = time.time()
    print("\n", "*" * 10, f"Run: {run_n}/{cfg.RUNS}", "*" * 10)

    # load data
    datasets = build_roleplay_dataset(cfg.LOADER)[0]
    datasets.pop(VALIDATION)
    datasets.pop(TEST)

    def bring_correct_option_forward(row):
        """
        Bring the correct option to the front of the option_texts list.
        """
        correct_option_index = (
            row["correct_option_id"] - 1
        )  # Convert to zero-based index
        row["option_texts"].insert(0, row["option_texts"].pop(correct_option_index))
        row["correct_option_id"] = (
            1  # Update correct option index to 1 (first position)
        )
        return row

    # apply function to bring correct option forward
    datasets[TRAIN] = datasets[TRAIN].apply(
        bring_correct_option_forward,
        axis=1,
    )
    # keep only questions with 4 answer options
    datasets[TRAIN] = datasets[TRAIN][datasets[TRAIN]["num_answer_options"] == 4]

    # subset
    if args.dry_run:
        logger.info("Dry run: using only 10 observations")
        datasets[TRAIN] = datasets[TRAIN].iloc[:10, :]

    # dataframes
    datasets_fmt = build_example_formatter(
        example_formatter_cfg=cfg.EXAMPLE_FORMATTER.QUESTIONS,
        datasets=datasets,
        is_interaction=False,
    )

    # list of dicts
    list_train = df_to_listdict(datasets_fmt[TRAIN])

    # seed
    set_seed(cfg.SEED + run_n)

    # structured output
    StrucOutput = build_structured_outputter(cfg.STRUCTURED_OUTPUTTER)

    # prompt
    prompt, _ = build_prompt(
        cfg=cfg,
        struc_output=StrucOutput,
        student_scale_str="",
        few_shot=False,
        q_ids_train=None,
        examples=None,
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
        data=list_train,
        prefix="train",
        structured=cfg.MODEL.NATIVE_STRUCTURED_OUTPUT,
        json_schema=StrucOutput,
        langfuse_handler=langfuse_handler,
    )

    # TODO: parse results and convert to good format

    write_pickle(  # TODO: dict
        {
            "preds_raw": {**val_preds_raw},
        },
        save_dir=os.path.join(cfg.OUTPUT_DIR, cfg.ID_MISCON),  # TODO: update path
        fname=f"run_{run_n}",
    )
    print_elapsed_time(start_time, run_n)
    langfuse_handler.flush()


def main() -> None:
    """Run experiment."""
    args = parser.parse_args()

    # config
    configs = load_configs(args.config, problem_type="misconceptions")

    # remove previous contents (take dir form first cfg)
    delete_previous_content(configs[0].OUTPUT_DIR)

    # logical checks before start running
    for cfg in configs:
        check_cfg(cfg, problem_type="misconceptions")

    # langfuse
    langfuse_session = Langfuse()

    errors = []
    for cfg in configs:

        print("\n", "=" * 10, f"Config: {cfg.ID_MISCON}", "=" * 10)
        # start experiment loop
        for run_n in range(1, cfg.RUNS + 1):
            run_single_cfg(
                cfg=cfg,
                run_n=run_n,
                args=args,
                langfuse_session=langfuse_session,
            )
        save_config(cfg, save_dir=cfg.OUTPUT_DIR, fname=cfg.ID_MISCON)
        # try:
        #     for run_n in range(1, cfg.RUNS + 1):
        #         run_single_cfg(
        #             cfg=cfg,
        #             run_n=run_n,
        #             args=args,
        #             langfuse_session=langfuse_session,
        #         )
        #     save_config(cfg, save_dir=cfg.OUTPUT_DIR, fname=cfg.ID_MISCON)
        # except Exception as e:
        #     errors.append((cfg.ID_MISCON, e))
        #     logger.error(
        #         "Error occurred during the experiment",
        #         config=cfg.ID_MISCON,
        #         error=str(e),
        #     )

    if len(errors) > 0:
        logger.error(
            "Errors occurred during the following experiments",
            configs=errors,
        )
    else:
        logger.info("All experiments completed successfully.")


if __name__ == "__main__":
    main()
