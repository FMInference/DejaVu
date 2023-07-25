import time
import json
import torch.nn.functional
from comm.comm_utils import *
from modules.generation_utils import get_logits_processor, get_logits_warper


from .dist_pipeline_inference_mask_sample_token_pipe_sync import DistSampleInferenceMaskTokenPipeSync


class DistSampleInferenceMaskTokenSparsePipeSync(DistSampleInferenceMaskTokenPipeSync):
    
    def _forward_compute_generate_token(self, index, mask=None):

        if mask is not None and self.num_completions > 1:
            # repeat n times
            mask = mask.repeat(self.num_completions, 1)

        # print("Compute generate seq micro-batch <", index, ">.")
        if self.pp_rank == 0:
            cache = self._get_cached_attention(0, index)
            current_emb = self.layers['emb'](self.recv_new_token[index], self.cached_attention[0][index], mask=mask)
        else:
            current_emb = self.input_token_emb[index]

        for layer_index in range(self.num_layers):

            cache = self._get_cached_attention(layer_index, index)

            if layer_index == 0:
                self.layers['block' + str(layer_index)].prepare_fc_weights(current_emb)
            if layer_index != self.num_layers -1:
                self.layers['block' + str(layer_index+1)].prepare_fc_weights(current_emb)

            current_emb, cache = \
                self.layers['block' + str(layer_index)](current_emb, cache, mask=mask)

            self._set_cached_attention(cache, layer_index, index)
        self.output_token_emb[index] = current_emb

        if self.pp_rank == self.pipeline_group_size - 1:
            self._generate_new_token(index)
