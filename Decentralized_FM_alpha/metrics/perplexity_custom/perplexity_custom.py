# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Perplexity Metric."""

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

import datasets
from datasets import logging


_CITATION = """\

"""

_DESCRIPTION = """
Perplexity (PPL) is one of the most common metrics for evaluating language models.
It is defined as the exponentiated average negative log-likelihood of a sequence.

For more information, see https://huggingface.co/docs/transformers/perplexity
"""

_KWARGS_DESCRIPTION = """
Args:
    model_id (str): model used for calculating Perplexity
            NOTE: Perplexity can only be calculated for causal language models.
                    This includes models such as gpt2, causal variations of bert,
                    causal versions of t5, and more (the full list can be found
                    in the AutoModelForCausalLM documentation here:
                    https://huggingface.co/docs/transformers/master/en/model_doc/auto#transformers.AutoModelForCausalLM )

    input_texts (list of str): input text, each separate text snippet
        is one list entry. Perplexity returned will be an average of
        the perplexity for each list entry.
    stride (int): stride size, defaults to 512
    device (str): device to run on, defaults to 'cuda' when available
Returns:
    perplexity: dictionary containing the average perplexity score for the text
        in the input list.
Examples:
    Example 1:
        >>> perplexity = datasets.load_metric("perplexity")
        >>> input_texts = ["lorem ipsum", "Happy Birthday!", "Bienvenue"]
        >>> results = perplexity.compute(model_id='gpt2',
        ...                              add_start_token=False,
        ...                              input_texts=input_texts) # doctest:+ELLIPSIS
        >>> print(list(results.keys()))
        ['perplexities', 'mean_perplexity']
        >>> print(round(results["mean_perplexity"], 2))
        78.22
        >>> print(round(results["perplexities"][0], 2))
        11.11

    Example 2:
        >>> perplexity = datasets.load_metric("perplexity")
        >>> input_texts = datasets.load_dataset("wikitext",
        ...                                     "wikitext-2-raw-v1",
        ...                                     split="test")["text"][:50] # doctest:+ELLIPSIS
        [...]
        >>> input_texts = [s for s in input_texts if s!='']
        >>> results = perplexity.compute(model_id='gpt2',
        ...                              input_texts=input_texts) # doctest:+ELLIPSIS
        >>> print(list(results.keys()))
        ['perplexities', 'mean_perplexity']
        >>> print(round(results["mean_perplexity"], 2))
        1977.55
        >>> print(round(results["perplexities"][0], 2))
        1349.56
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PerplexityCustom(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float32"),
                    "references": datasets.Sequence(datasets.Value("float32")),
                }
            ),
        )

    def _compute(self, predictions, references):
        
        loss = torch.tensor(predictions).mean()
        
        ppls = torch.exp(loss)
        
        return {"perplexity": ppls.item(), "loss": loss.item()}