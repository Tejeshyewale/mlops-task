#!/usr/bin/env python3
"""
run.py - Minimal MLOps-style batch job.

Loads config from YAML, reads OHLCV data, computes a rolling mean on
`close`, generates a binary signal, and writes structured metrics
JSON along with detailed logs.

Usage:
    python run.py --input data.csv --config config.yaml \
                   --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REQUIRED_CONFIG_FIELDS = ["seed", "window", "version"]
REQUIRED_COLUMN = "close"


def parse_args():
    parser = argparse.ArgumentParser(description="MLOps batch job: rolling mean signal generator")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", required=True, help="Path to write metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to write log file")
    return parser.parse_args()


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_task")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
    )

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger


def write_metrics(output_path: str, payload: dict, logger: logging.Logger):
    try:
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Metrics written to {output_path}")
    except Exception as e:
        # Last-resort: log failure to write metrics itself.
        logger.error(f"Failed to write metrics file: {e}")


def load_and_validate_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")

    if not isinstance(config, dict):
        raise ValueError("Invalid config structure: expected a YAML mapping/object")

    missing = [field for field in REQUIRED_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Missing required config field(s): {', '.join(missing)}")

    if not isinstance(config["seed"], int):
        raise ValueError("Config field 'seed' must be an integer")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError("Config field 'window' must be a positive integer")
    if not isinstance(config["version"], str):
        raise ValueError("Config field 'version' must be a string")

    logger.info(
        f"Config loaded and validated: seed={config['seed']}, "
        f"window={config['window']}, version={config['version']}"
    )
    return config


def load_and_validate_data(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise ValueError(f"Input file not found: {input_path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file is empty or has no columns: {input_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV format: {e}")
    except Exception as e:
        raise ValueError(f"Failed to read input file: {e}")

    if df.empty:
        raise ValueError("Input file contains no rows")

    if len(df.columns) == 1 and "," in df.columns[0]:
        logger.warning(
            "Detected malformed CSV (entire row parsed as a single quoted "
            "field). Attempting to recover by re-splitting on commas."
        )
        try:
            with open(path, "r") as f:
                lines = [line.strip().strip('"') for line in f.readlines() if line.strip()]
            from io import StringIO
            df = pd.read_csv(StringIO("\n".join(lines)))
            logger.info(f"Recovered CSV structure: columns={df.columns.tolist()}")
        except Exception as e:
            raise ValueError(f"Invalid CSV format (could not recover from malformed rows): {e}")

    if REQUIRED_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: '{REQUIRED_COLUMN}'")

    if not pd.api.types.is_numeric_dtype(df[REQUIRED_COLUMN]):
        try:
            df[REQUIRED_COLUMN] = pd.to_numeric(df[REQUIRED_COLUMN], errors="raise")
        except Exception:
            raise ValueError(f"Column '{REQUIRED_COLUMN}' contains non-numeric data")

    logger.info(f"Rows loaded: {len(df)}")
    return df


def compute_rolling_mean_and_signal(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Computing rolling mean on 'close' with window={window}")
    # First (window-1) rows will have NaN rolling mean; this is expected
    # and handled consistently by excluding them from signal computation.
    df["rolling_mean"] = df[REQUIRED_COLUMN].rolling(window=window, min_periods=window).mean()

    logger.info("Generating binary signal: 1 if close > rolling_mean else 0")
    df["signal"] = np.where(df[REQUIRED_COLUMN] > df["rolling_mean"], 1, 0)
    # Rows without a valid rolling mean (first window-1 rows) get signal = 0
    # by definition of the comparison (NaN comparisons are False -> 0),
    # and are documented as not eligible for a "real" signal.
    df.loc[df["rolling_mean"].isna(), "signal"] = 0

    return df


def main():
    args = parse_args()
    logger = setup_logging(args.log_file)
    start_time = time.time()

    logger.info("=== Job started ===")
    logger.info(
        f"Args: input={args.input}, config={args.config}, "
        f"output={args.output}, log_file={args.log_file}"
    )

    version_for_error = "unknown"

    try:
        config = load_and_validate_config(args.config, logger)
        version_for_error = config.get("version", "unknown")

        np.random.seed(config["seed"])
        logger.info(f"Random seed set to {config['seed']}")

        df = load_and_validate_data(args.input, logger)

        df = compute_rolling_mean_and_signal(df, config["window"], logger)

        rows_processed = len(df)
        signal_rate = float(df["signal"].mean())

        latency_ms = int(round((time.time() - start_time) * 1000))

        metrics = {
            "version": config["version"],
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": latency_ms,
            "seed": config["seed"],
            "status": "success",
        }

        logger.info(
            f"Metrics summary: rows_processed={rows_processed}, "
            f"signal_rate={metrics['value']}, latency_ms={latency_ms}"
        )

        write_metrics(args.output, metrics, logger)
        logger.info("=== Job ended: status=success ===")
        print(json.dumps(metrics, indent=2))
        sys.exit(0)

    except Exception as e:
        latency_ms = int(round((time.time() - start_time) * 1000))
        error_message = str(e)
        logger.error(f"Job failed: {error_message}", exc_info=True)

        error_metrics = {
            "version": version_for_error,
            "status": "error",
            "error_message": error_message,
        }

        write_metrics(args.output, error_metrics, logger)
        logger.info("=== Job ended: status=error ===")
        print(json.dumps(error_metrics, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
