"""Constants."""

import re
from functools import partial
from typing import List

INPUT_COL = "text"
OUTPUT_COL = "label_str"
DATASET_COL = "dataset"

def split_choices_on_keyword(query: str, choice_str: str) -> List[str]:
    """Extract gold choices from query."""
    if choice_str not in query:
        return []
    all_choices = query.split(choice_str)[1:]
    all_choices = [c.strip().split("\n")[0] for c in all_choices]
    return all_choices


def regex_search_choices(query: str, regex_pattern: re.Pattern) -> List[str]:
    """Extract choices from query using regex."""
    matches = regex_pattern.search(query)
    choices = [c.strip() for c in matches.groups()]
    return choices


from transformers import (
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoForCausalLM,
    OPTForCausalLM,
)

# from fm_in_context_eval.fine_tuning.models.gpt_adaptor import (
#     GPTJAdaptorForCausalLM,
#     GPTNeoAdaptorForCausalLM,
# )

# MODEL_REGISTRY = {
#     "gpt_adaptor-125M": GPTNeoAdaptorForCausalLM,
#     "gpt_adaptor-1.3B": GPTNeoAdaptorForCausalLM,
#     "gpt_adaptor-2.7B": GPTNeoAdaptorForCausalLM,
#     "gpt_adaptor-6B": GPTJAdaptorForCausalLM,
#     "EleutherAI/gpt-j-6B": GPTJForCausalLM,
#     "EleutherAI/gpt-neo-125M": GPTNeoForCausalLM,
#     "EleutherAI/gpt-neo-1.3B": GPTNeoForCausalLM,
#     "EleutherAI/gpt-neo-2.7B": GPTNeoForCausalLM,
#     "facebook/opt-125m": OPTForCausalLM,
#     "facebook/opt-355m": OPTForCausalLM,
#     "facebook/opt-1.3b": OPTForCausalLM,
#     "facebook/opt-2.7b": OPTForCausalLM,
#     "facebook/opt-6.7b": OPTForCausalLM,
#     "facebook/opt-13b": OPTForCausalLM,
#     "facebook/opt-30b": OPTForCausalLM,
#     "gpt2": GPT2LMHeadModel,
# }

# TOK_CONFIG_MAPPING = {
#     "gpt_adaptor-125M": "EleutherAI/gpt-neo-125M",
#     "gpt_adaptor-1.3B": "EleutherAI/gpt-neo-1.3B",
#     "gpt_adaptor-2.7B": "EleutherAI/gpt-neo-2.7B",
#     "gpt_adaptor-6B": "EleutherAI/gpt-j-6B",
# }

# try:
#     from fm_in_context_eval.fine_tuning.models.gpt_flash_neo import (
#         GPTNeoForCausalLM as GPTNeoForCausalLMFlash,
#     )

#     MODEL_REGISTRY["EleutherAI/gpt-neo-flash-1.3B"] = GPTNeoForCausalLMFlash
#     TOK_CONFIG_MAPPING["EleutherAI/gpt-neo-flash-1.3B"] = "EleutherAI/gpt-neo-1.3B"
# except:
#     print("Flash attention not installed.")
#     pass


TASK2METRICS = {
    "conll": ["accuracy"],
    "MMLU.*": ["multi-choice"],
    "MMLU_humanities": ["multi-choice"],
    "MMLU_STEM": ["multi-choice"],
    "MMLU_social_sciences": ["multi-choice"],
    "MMLU_other": ["multi-choice"],
    "backpage": ["accuracy"],
    "story_cloze": ["accuracy", "rouge"],
    "entity_imputation_Buy": ["accuracy"],
    "entity_imputation_Restaurant": ["accuracy"],
    "entity_error_detection_Hospital": ["binary_f1"],
    "entity_matching_Beer": ["binary_f1"],
    "entity_matching_iTunes_Amazon": ["binary_f1"],
    "entity_matching_Fodors_Zagats": ["binary_f1"],
    "entity_matching_DBLP_ACM": ["binary_f1"],
    "entity_matching_DBLP_GoogleScholar": ["binary_f1"],
    "entity_matching_Amazon_Google": ["binary_f1"],
    "entity_matching_Walmart_Amazon": ["binary_f1"],
    "bigbench_code_line_description": ["accuracy", "rouge"],
    "bigbench_hindu_knowledge": ["accuracy"],
    "bigbench_known_unknowns": ["accuracy"],
    "bigbench_language_identification": ["accuracy"],
    "bigbench_logic_grid_puzzle": ["accuracy"],
    "bigbench_logical_deduction": ["accuracy"],
    "bigbench_misconceptions": ["accuracy"],
    "bigbench_movie_dialog_same_or_different": ["accuracy"],
    "bigbench_novel_concepts": ["accuracy", "rouge"],
    "bigbench_strategyqa": ["accuracy"],
    "bigbench_formal_fallacies_syllogisms_negation": ["accuracy"],
    "bigbench_vitaminc_fact_verification": ["accuracy"],
    "bigbench_winowhy": ["accuracy"],
    "bigbench_conceptual_combinations": ["accuracy", "rouge"],
    "instructeval_cnn_dm": ["rouge"],
    "instructeval_drop": ["f1_text"],
    "instructeval_fr_to_en": ["rouge"],
    "instructeval_quac": ["f1_text"],
    "instructeval_squadv2": ["f1_text"],
    "instructeval_tldr": ["rouge"],
    "hellaswag": ["accuracy"],
    "winogrande_winogrande_xl": ["accuracy", "rouge"],
    "anli": ["accuracy"],
    "super_glue_cb": ["accuracy"],
    "super_glue_copa": ["accuracy"],
    "super_glue_wic": ["accuracy"],
    "super_glue_rte": ["accuracy"],
    "super_glue_wsc.fixed": ["accuracy"],
    "tatqa": ["accuracy"],
}

# https://github.com/bigscience-workshop/promptsource/tree/main/promptsource/templates
TASKPROMPT2LAMBDACHOICES = {
    ("bigbench_code_line_description", None): partial(
        split_choices_on_keyword, choice_str="choice:"
    ),
    ("bigbench_hindu_knowledge", None): partial(
        split_choices_on_keyword, choice_str="choice:"
    ),
    ("bigbench_known_unknowns", None): partial(
        split_choices_on_keyword, choice_str="choice:"
    ),
    ("bigbench_language_identification", None): partial(
        split_choices_on_keyword, choice_str="choice:"
    ),
    ("bigbench_logic_grid_puzzle", None): partial(
        split_choices_on_keyword, choice_str="choice:"
    ),
    ("bigbench_conceptual_combinations", None): partial(
        split_choices_on_keyword, choice_str="option:"
    ),
    ("winogrande_winogrande_xl", "does_underscore_refer_to"): partial(
        regex_search_choices,
        regex_pattern=re.compile(
            r"In the previous sentence, does _ refer to (.*) or (.*)\?"
        ),
    ),
    ("super_glue_copa", "_why_C1_or_C2"): partial(
        regex_search_choices, regex_pattern=re.compile(r'Why\? "(.*)" or "(.*)"\?')
    ),
}

TASKPROMPT2FIXEDCHOICES = {
    ("MMLU.*", None): ["A", "B", "C", "D"],
    ("MMLU_humanities", None): ["A", "B", "C", "D"],
    ("MMLU_STEM", None): ["A", "B", "C", "D"],
    ("MMLU_social_sciences", None): ["A", "B", "C", "D"],
    ("MMLU_other", None): ["A", "B", "C", "D"],
    ("story_cloze", None): ["A", "B"],
    ("entity_error_detection_Hospital", None): ["Yes", "No"],
    ("entity_matching_Beer", None): ["Yes", "No"],
    ("entity_matching_iTunes_Amazon", None): ["Yes", "No"],
    ("entity_matching_Fodors_Zagats", None): ["Yes", "No"],
    ("entity_matching_DBLP_ACM", None): ["Yes", "No"],
    ("entity_matching_DBLP_GoogleScholar", None): ["Yes", "No"],
    ("entity_matching_Amazon_Google", None): ["Yes", "No"],
    ("entity_matching_Walmart_Amazon", None): ["Yes", "No"],
    ("bigbench_misconceptions", None): ["true", "false"],
    ("bigbench_movie_dialog_same_or_different", None): ["same", "different"],
    ("bigbench_strategyqa", None): ["Yes", "No"],
    ("bigbench_formal_fallacies_syllogisms_negation", None): ["invalid", "valid"],
    ("bigbench_vitaminc_fact_verification", None): ["True", "False", "Neither"],
    ("bigbench_winowhy", None): ["correct", "incorrect"],
    ("anli", "justified_in_saying_r3"): ["Maybe", "Yes", "No"],
    ("hellaswag", "if_begins_how_continues"): [
        "Ending 1",
        "Ending 2",
        "Ending 3",
        "Ending 4",
    ],
    ("super_glue_cb", "claim_true_false_inconclusive"): [
        "False",
        "True",
        "Inconclusive",
    ],
    ("super_glue_wic", "affirmation_true_or_false"): ["True", "False"],
    ("super_glue_rte", "GPT_3_style"): ["False", "True"],
    ("super_glue_wsc.fixed", "GPT_3_Style"): ["Yes", "No"],
}

T0_TASKS = {
    "adversarial_qa",
    "ag_news",
    "ai2_arc",
    "amazon_polarity",
    "anli",
    "app_reviews",
    "cnn_dailymail",
    "common_gen",
    "cos_e_v1.11",
    "cosmos_qa",
    "dbpedia_14",
    "dream",
    "duorc",
    "gigaword",
    "glue_mrpc",
    "glue_qqp",
    "hellaswag",
    "imdb",
    "kilt",
    "multi_news",
    "openbookqa",
    "paws",
    "piqa",
    "qasc",
    "quail",
    "quarel",
    "quarel",
    "quartz",
    "quoref",
    "race",
    "ropes",
    "rotten_tomatoes",
    "samsum",
    "sciq",
    "social_i_qa",
    "squad_v2",
    "super_glue_boolq",
    "super_glue_cb",
    "super_glue_copa",
    "super_glue_multirc",
    "super_glue_record",
    "super_glue_rte",
    "super_glue_wic",
    "super_glue_wsc.fixed",
    "trec",
    "trivia_qa",
    "web_questions",
    "wiki_bio",
    "wiki_hop",
    "wiki_qa",
    "winogrande_winogrande_debiased",
    "winogrande_winogrande_xl",
    "wiqa",
    "xsum",
    "yelp_review",
}
