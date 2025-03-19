"""Module for fine-tuning."""

# standard library imports
import argparse
import os
import time

# related third party imports
import structlog
import numpy as np

# local application/library specific imports
from data_loader.build import build_dataset
from tools.configurator import check_cfg, load_configs, save_config
from tools.utils import (
    delete_previous_content,
    print_elapsed_time,
    write_pickle,
    set_seed,
    BatchCallback,
    format_time,
)
from prompt.few_shot_prompt import (
    df_to_listdict,
    human_format_input,
    human_format_output,
    apply_prompt_fmt,
)
from tools.constants import TRAIN, VALIDATION, TEST
from prompt.build import build_prompt
from prompt.json_schema import MCQAnswer, validate_output
from model.build import build_model
from tools.metrics import compute_metrics

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
            start_time = time.time()
            print("\n", "*" * 10, f"Run: {run_n}/{cfg.RUNS}", "*" * 10)

            # load data
            dataset = build_dataset(cfg.LOADER, cfg.SEED + run_n)
            # subset
            # TODO: remove for real run!
            # dataset[VALIDATION] = dataset[VALIDATION].iloc[:10, :]
            if args.dry_run:
                logger.info("Dry run: using only 10 validation observations")
                dataset[VALIDATION] = dataset[VALIDATION].iloc[:10, :]

            # dataframes
            df_train = apply_prompt_fmt(
                df=dataset[TRAIN],
                input_fmt=human_format_input,  # TODO: make dependent on dataset (e.g., "reading context" for CUPA)
                output_fmt=human_format_output,
            )
            df_val = apply_prompt_fmt(
                df=dataset[VALIDATION],
                input_fmt=human_format_input,
                output_fmt=human_format_output,
            )
            df_test = apply_prompt_fmt(
                df=dataset[TEST],
                input_fmt=human_format_input,
                output_fmt=human_format_output,
            )

            # list of dicts
            list_train = df_to_listdict(df_train)
            list_val = df_to_listdict(df_val)
            list_test = df_to_listdict(df_test)  # noqa

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
            logger.info("Predict - start")
            pred_start_time = time.time()
            cb = BatchCallback(len(list_val))  # init callback
            preds_raw = chain.batch(list_val, config={"callbacks": [cb]})
            cb.progress_bar.close()
            if cfg.MODEL.STRUCTURED_OUTPUT:
                # get all raw outputs
                preds_raw = [output["raw"] for output in preds_raw]
            preds_validated = validate_output(preds_raw, schema=MCQAnswer)
            pred_time = time.time() - pred_start_time
            logger.info("Predict - end", time=format_time(pred_time))

            # evaluate
            logger.info("Evaluate - start")
            y_val_pred = np.array([output.student_answer for output in preds_validated])
            y_val_student = dataset[VALIDATION]["student_answer"].to_numpy()
            y_val_true = dataset[VALIDATION]["correct_answer"].to_numpy()
            metrics = compute_metrics(
                y_val_pred=y_val_pred,
                y_val_true=y_val_true,
                y_val_student=y_val_student,
            )
            # TODO: also do with test set
            logger.info("Evaluate - end", accuracy=metrics["acc_student_pred"])

            write_pickle(
                {
                    "metrics": metrics,
                    "preds_raw": preds_raw,
                    "preds_validated": preds_validated,
                    "y_pred": y_val_pred,
                    "y_true": y_val_true,
                    "y_student": y_val_student,
                    "pred_time": pred_time,
                },
                save_dir=os.path.join(cfg.OUTPUT_DIR, cfg.ID),
                fname=f"run_{run_n}",
            )
            print_elapsed_time(start_time, run_n)

        save_config(cfg, save_dir=cfg.OUTPUT_DIR, fname=cfg.ID)


if __name__ == "__main__":
    main()
