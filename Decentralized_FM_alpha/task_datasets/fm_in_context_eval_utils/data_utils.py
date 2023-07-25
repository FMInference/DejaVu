"""Data utils."""
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .constants import DATASET_COL, INPUT_COL, OUTPUT_COL

logger = logging.getLogger(__name__)


def sample_train_data(train: pd.DataFrame, n_rows: int, random_state: int = None):
    """
    Sample train data.

    Used when random sampling points for prompt.
    """
    res = train.sample(n_rows, random_state=random_state, replace=False)
    return res


def read_other_single(
    split_path: str,
) -> pd.DataFrame:
    """Read col and assert label and text columns exist."""
    data = pd.read_feather(split_path)
    assert INPUT_COL in data.columns
    assert DATASET_COL in data.columns
    if OUTPUT_COL not in data.columns:
        data[OUTPUT_COL] = ""
    else:
        data[OUTPUT_COL] = data[OUTPUT_COL].astype(str)
        # Add space if no starting whitespace
        no_leading_whitespace = ~data[OUTPUT_COL].str.startswith((" ", "\n"))
        data.loc[no_leading_whitespace, OUTPUT_COL] = (
            " " + data.loc[no_leading_whitespace, OUTPUT_COL]
        )
        data[OUTPUT_COL] = data[OUTPUT_COL] + "\n"
    return data


def read_raw_data(
    data_dir: str,
):
    """Read in data where each directory is unique for a task."""
    data_files_sep = {}
    logger.info(f"Processing {data_dir}")
    task = "other"
    data_dir_p = Path(data_dir)
    if task == "other":
        train_file = data_dir_p / "train_k0.feather"
        valid_file = data_dir_p / "validation_k0.feather"
        test_file = data_dir_p / "test_k0.feather"
        read_data_func = read_other_single
        label_col = "label_str"
    else:
        raise ValueError(f"Task {task} not recognized.")

    data_files_sep["train"] = read_data_func(train_file)
    # Read validation
    if valid_file.exists():
        data_files_sep["validation"] = read_data_func(valid_file)
    # Read test
    if test_file.exists():
        data_files_sep["test"] = read_data_func(test_file)
    return data_files_sep, label_col
