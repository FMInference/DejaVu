'''
adapted from huggingface
https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/generation_utils.py#L62
'''

import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.generation_beam_constraints import Constraint, DisjunctiveConstraint, PhrasalConstraint
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)



def get_logits_processor(
    repetition_penalty: float = None,
    no_repeat_ngram_size: int = None,
    input_ids_seq_length: int = None,
    bad_words_ids: List[List[int]] = None,
    min_length: int = None,
    max_length: int = None,
    eos_token_id: int = None,
    forced_bos_token_id: int = None,
    forced_eos_token_id: int = None,
    prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]] = None,
    num_beams: int = None,
    num_beam_groups: int = None,
    diversity_penalty: float = None,
    remove_invalid_values: bool = None,
    exponential_decay_length_penalty: Tuple = None,
    renormalize_logits: Optional[bool] = None,
) -> LogitsProcessorList:
    
    processors = LogitsProcessorList()
    
    # instantiate processors list

    # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
    # all samplers can be found in `generation_utils_samplers.py`
    if diversity_penalty is not None and diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
            )
        )
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if bad_words_ids is not None:
        processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
    if min_length is not None and eos_token_id is not None and min_length > 0:
        processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
    if prefix_allowed_tokens_fn is not None:
        processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams // num_beam_groups))
    if forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
    if forced_eos_token_id is not None:
        processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
    if remove_invalid_values is True:
        processors.append(InfNanRemoveLogitsProcessor())
    if exponential_decay_length_penalty is not None:
        processors.append(
            ExponentialDecayLengthPenalty(exponential_decay_length_penalty, eos_token_id, input_ids_seq_length)
        )
        
    if renormalize_logits is True:
        processors.append(LogitNormalization())
    return processors



def get_logits_warper(
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    typical_p: Optional[float] = None,
    temperature: Optional[float] = None,
    num_beams: Optional[int] = None,
    renormalize_logits: Optional[bool] = None,
) -> LogitsProcessorList:
    """
    This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
    used for multinomial sampling.
    """
    # instantiate warpers list
    warpers = LogitsProcessorList()
    # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
    # all samplers can be found in `generation_utils_samplers.py`
    if temperature is not None and temperature != 1.0:
        if temperature == 0.0:
            print(f"temperature is 0, should be deterministic (greedy).")
            top_k = 1
        else:
            warpers.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
    if top_p is not None and top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
    if typical_p is not None and typical_p < 1.0:
        warpers.append(TypicalLogitsWarper(mass=typical_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
    # `LogitNormalization` should always be the last logit processor, when present
    if renormalize_logits is True:
        warpers.append(LogitNormalization())
    return warpers


