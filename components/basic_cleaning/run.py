import argparse
import logging
import wandb
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(vars(args))

    logger.info(f"Downloading artifact {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)

    logger.info("Dropping outliers based on price")
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("Converting last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    logger.info("Saving cleaned data to local file")
    clean_path = Path("clean_sample.csv")
    df.to_csv(clean_path, index=False)

    logger.info("Logging cleaned data as W&B artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(clean_path)
    run.log_artifact(artifact)

    logger.info("Basic cleaning completed")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the raw data artifact to clean",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the cleaned data artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the cleaned data artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the cleaned data artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum acceptable price for listings",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum acceptable price for listings",
        required=True,
    )

    args = parser.parse_args()

    go(args)
