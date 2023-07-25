import logging
import random
from collections import defaultdict
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Dict, List, Union
from .constants import DATASET_COL, INPUT_COL, OUTPUT_COL

import pandas as pd
import transformers
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import PretrainedConfig
from transformers.testing_utils import CaptureLogger

from .data_utils import read_raw_data, sample_train_data

logger = logging.getLogger(__name__)


def update_config_from_string(config: PretrainedConfig, update_str: str):
    """
    Updates attributes of this class with attributes from `update_str`.

    The expected format is ints, floats and strings as is, and for booleans use `true` or `false`. For example:
    "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"

    The keys to change have to already exist in the config object.

    Args:
        update_str (`str`): String with attributes that should be updated for this class.

    """

    d = dict(x.split("=") for x in update_str.split(","))
    for k, v in d.items():
        if not hasattr(config, k):
            raise ValueError(f"key {k} isn't in the original config dict")

        old_v = getattr(config, k)
        if isinstance(old_v, bool):
            if v.lower() in ["true", "1", "y", "yes"]:
                v = True
            elif v.lower() in ["false", "0", "n", "no"]:
                v = False
            else:
                raise ValueError(f"can't derive true or false from {v} (key {k})")
        elif isinstance(old_v, float) or old_v == 0:
            v = float(v)
        elif isinstance(old_v, int):
            v = int(v)
        elif not isinstance(old_v, str):
            raise ValueError(
                f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
            )

        setattr(config, k, v)


def merge_datasets(datasetA: DatasetDict, datasetB: DatasetDict):
    # Join datasets
    all_datasets = {}
    for k in set(datasetA.keys()).union(set(datasetB.keys())):
        if k in datasetA and k in datasetB:
            all_datasets[k] = concatenate_datasets([datasetA[k], datasetB[k]])
        elif k in datasetA:
            all_datasets[k] = datasetA[k]
        elif k in datasetB:
            all_datasets[k] = datasetB[k]
        else:
            raise ValueError(
                f"Key {k} not in either datasetA or hf_datasets_without_label"
            )
    return DatasetDict(all_datasets)


def read_data(
    data_dirs: List[str],
    class_balanced: bool = False,
    max_train_samples: int = -1,
    max_train_percent: float = -1,
    local_rank: int = -1,
):
    """Read in data where each directory is unique for a task."""
    if max_train_samples > 0 and max_train_percent > 0:
        raise ValueError(
            "Only one of max_train_samples and max_train_percent can be specified"
        )
    if max_train_percent > 1.0:
        raise ValueError("max_train_percent must be between 0 and 1")
    data_files_with_label = defaultdict(dict)
    data_files_without_label = defaultdict(dict)
    for data_dir in data_dirs:
        data_files_sep_single, label_col = read_raw_data(
            data_dir=data_dir,
        )
        # If there is no label col
        if data_files_sep_single["train"][label_col].str.strip().str.len().sum() == 0:
            dict_to_add = data_files_without_label
        else:
            dict_to_add = data_files_with_label
        # Don't class balance on open ended classificiation tasks
        if class_balanced:
            # Class balance sample the train data
            label_cnts = data_files_sep_single["train"].groupby(label_col).count()
            sample_per_class = label_cnts.min()[INPUT_COL]
            print(f"Train sample per class: {sample_per_class}")
            data_files_sep_single["train"] = (
                data_files_sep_single["train"]
                .groupby(label_col, group_keys=False)
                .apply(lambda x: x.sample(sample_per_class, random_state=42))
            )

        # Shuffle train data
        data_files_sep_single["train"] = (
            data_files_sep_single["train"]
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

        # Sample data
        orig_train_len = len(data_files_sep_single["train"])
        if max_train_percent > 0:
            max_examples = int(max_train_percent * orig_train_len)
        elif max_train_samples > 0:
            max_examples = min(orig_train_len, max_train_samples)
        else:
            max_examples = orig_train_len
        data_files_sep_single["train"] = data_files_sep_single["train"].iloc[
            :max_examples
        ]

        print(
            f"Length of {data_dir} train is {data_files_sep_single['train'].shape[0]} from {orig_train_len}"
        )
        dict_to_add["train"][data_dir] = data_files_sep_single["train"]

        if "validation" in data_files_sep_single:
            if data_files_sep_single["validation"].shape[0] > 0:
                dict_to_add["validation"][data_dir] = data_files_sep_single[
                    "validation"
                ]
        if "test" in data_files_sep_single:
            if data_files_sep_single["test"].shape[0] > 0:
                dict_to_add["test"][data_dir] = data_files_sep_single["test"]

    # Convert to HF datasets
    def map_to_hf(data_files: Dict[str, pd.DataFrame]):
        hf_datasets = {}
        for k in data_files:
            dataset = pd.concat(data_files[k].values()).reset_index(drop=True)
            to_print = dataset.sample(3)
            if local_rank in [0, -1]:
                for i in range(len(to_print)):
                    print(
                        "DATASET:\n*******\n",
                        to_print.iloc[i][DATASET_COL],
                        "\n*******\n",
                        "INPUT:\n*******\n",
                        to_print.iloc[i][INPUT_COL],
                        "\n*******\n",
                        "OUTPUT:\n*******\n",
                        to_print.iloc[i][OUTPUT_COL],
                        "\n*******\n",
                    )
            # Sort for validation
            if k != "train":
                dataset = dataset.sort_values(by=DATASET_COL).reset_index(drop=True)
            # Hacky fix to get around pyarrow error
            for c in dataset:
                dataset[c] = dataset[c].astype(str)
            hf_datasets[k] = Dataset.from_pandas(
                dataset[[DATASET_COL, INPUT_COL, OUTPUT_COL]]
            )
        return hf_datasets

    hf_datasets_with_label = map_to_hf(data_files_with_label)
    hf_datasets_without_label = map_to_hf(data_files_without_label)
    if local_rank in [0, -1]:
        for k in hf_datasets_with_label:
            print(f"{k} WITH LABEL {len(hf_datasets_with_label[k])}")
        for k in hf_datasets_without_label:
            print(f"{k} NO LABEL {len(hf_datasets_without_label[k])}")
    hf_datasets_with_label = DatasetDict(hf_datasets_with_label)
    hf_datasets_without_label = DatasetDict(hf_datasets_without_label)
    return hf_datasets_with_label, hf_datasets_without_label


def tokenize_data(
    hf_datasets: DatasetDict,
    tokenizer: transformers.PreTrainedTokenizer,
    preprocessing_num_workers: int,
    overwrite_cache: bool,
    ignore_label_column: bool,
    data_cache_dir: str,
    block_size: int,
    mask_input: bool,
    training_args,
):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if "train" in hf_datasets:
        column_names = hf_datasets["train"].column_names
    else:
        column_names = hf_datasets["validation"].column_names
    text_column_name = INPUT_COL if INPUT_COL in column_names else column_names[0]
    label_column_name = OUTPUT_COL if OUTPUT_COL in column_names else column_names[1]
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    # Used for LLM inputs where we do group_blocks
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    # Used for prompting inputs that we do not group_blocks
    def tokenize_no_group_function(examples, split, mask_input):
        with CaptureLogger(tok_logger) as cl:
            # if group_blocks is True:
            #     raise ValueError("group_blocks should be False when calling tokenize_no_group_function for prompting inputs")
            output = defaultdict(list)
            number_overflow = 0
            output["labels"] = []
            # print(len(examples))
            for i in range(len(examples[label_column_name])):
                output_label = tokenizer(
                    examples[label_column_name][i],
                    padding=False,
                    truncation=True,
                    max_length=block_size,
                )
                overflow_check = tokenizer(
                    examples[text_column_name][i],
                    padding=False,
                    truncation=True,
                    return_overflowing_tokens=True,
                    # Leave room for the answer
                    max_length=(block_size - len(output_label["input_ids"])),
                )
                if "overflow_to_sample_mapping" in overflow_check:
                    overflow_mapping = overflow_check["overflow_to_sample_mapping"]
                    number_overflow += min(len(overflow_mapping) - 1, 1)

                output_inp = tokenizer(
                    examples[text_column_name][i],
                    padding=False,
                    truncation=True,
                    return_overflowing_tokens=False,
                    # Leave room for the answer
                    max_length=(block_size - len(output_label["input_ids"])),
                )
                # Mask input is mask_input True or test data
                if mask_input or (split != "train"):
                    output["labels"].append(
                        [-100] * len(output_inp["input_ids"])
                        + output_label["input_ids"]
                    )
                else:
                    output["labels"].append(
                        output_inp["input_ids"] + output_label["input_ids"]
                    )
                output["labels"][-1] += [-100] * (
                    block_size - len(output["labels"][-1])
                )
                for k in output_inp:
                    output[k].append(output_inp[k] + output_label[k])
                    if k == "input_ids":
                        output[k][-1] += [tokenizer.pad_token_id] * (
                            block_size - len(output[k][-1])
                        )
                    elif k == "attention_mask":
                        output[k][-1] += [0] * (block_size - len(output[k][-1]))
                    else:
                        raise ValueError(f"{k} not recognized")
                    

        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        print(
            f"Overflow Amount!!! {number_overflow} {100*number_overflow/len(examples[text_column_name]):.1f}%"
        )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = {}
        # Extract by split so we can pass it into the tokenizer
        for split in hf_datasets:
            if ignore_label_column:
                if split == "train":
                    tokenize_func = tokenize_function
                else:
                    # When ignore label column, never mask input but we keep
                    # the no_group_function so we can use our index offset into
                    # slices for our sliced evaluation
                    tokenize_func = partial(
                        tokenize_no_group_function, split=split, mask_input=False
                    )
            else:
                if split == "train":
                    tokenize_func = partial(
                        tokenize_no_group_function, split=split, mask_input=mask_input
                    )
                else:
                    # Always mask input for validation data when not ignore column
                    tokenize_func = partial(
                        tokenize_no_group_function, split=split, mask_input=True
                    )

            '''
            cache_file_name = (
                Path(data_cache_dir)
                / f"cached_tok_{int(ignore_label_column)}ig"
                / f"processed_{split}.dataset"
            )
            cache_file_name.parent.mkdir(exist_ok=True, parents=True)
            '''
            tokenized_datasets[split] = hf_datasets[split].map(
                tokenize_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=column_names,
                #load_from_cache_file=not overwrite_cache,
                #cache_file_name=str(cache_file_name),
                desc=f"Running tokenizer on dataset {split}",
            )
        tokenized_datasets = DatasetDict(tokenized_datasets)
        if training_args.local_rank in [-1, 0]:
            for i in range(2):
                print(tokenized_datasets["train"][i])
                print("*****************")

    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({block_size}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)

    if ignore_label_column:
        validation_ds = None
        if "validation" in tokenized_datasets:
            validation_ds = tokenized_datasets["validation"]
            del tokenized_datasets["validation"]

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.

        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = {}
            for split in tokenized_datasets:
                '''
                cache_file_name = (
                    Path(data_cache_dir)
                    / f"cached_grp_{int(ignore_label_column)}ig"
                    / f"processed_{split}.dataset"
                )
                cache_file_name.parent.mkdir(exist_ok=True, parents=True)
                '''

                lm_datasets[split] = tokenized_datasets[split].map(
                    group_texts,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    #load_from_cache_file=not overwrite_cache,
                    #cache_file_name=str(cache_file_name),
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            lm_datasets = DatasetDict(lm_datasets)
            if training_args.local_rank in [-1, 0]:
                for i in range(2):
                    print(lm_datasets["train"][i])
                    print("*****************")

        # don't group validation
        if validation_ds is not None:
            lm_datasets["validation"] = validation_ds
    else:  # No group blocking
        lm_datasets = tokenized_datasets
    return lm_datasets
