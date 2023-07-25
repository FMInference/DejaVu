import numpy as np
import torch
from torch import nn
from .hf_gptneo_train_module import GPTEmbeddings, GPTBlock, GPTClassificationHead, GPTLMHead
from comm.comm_utils import *

from copy import deepcopy


class GPTStageBase(nn.Module):
    def __init__(self, args, config):
        super(GPTStageBase, self).__init__()
        self._to_cpu = (args.dist_backend == "gloo")
        self._embedding_dim = args.embedding_dim  # embedding dimension
        self._seq_length = args.seq_length
        # the dimension of the feedforward aws_network model in nn.TransformerEncoder
        self._feedforward_dim = args.embedding_dim * 4
        self._num_heads = args.num_heads  # the number of heads in the multi-head attention models
        self._num_layers = args.num_layers
        self._layer_begin = get_pipeline_parallel_rank() * args.num_layers
        self._layer_end = min(self._layer_begin + args.num_layers, args.max_layers)
        
        self._task_type = getattr(args, 'task_type', 'classification')
        
        self.load_pretrained_model = args.load_pretrained_model
        self.model_name = args.model_name
        self.config = config

    def _create_first_layer(self):
        layer = GPTEmbeddings(deepcopy(self.config))
        if self.load_pretrained_model:
            layer.load_state_dict(
                torch.load(f'{self.model_name}/pytorch_embs.pt')
            )
        return layer

    def _create_last_layer(self):
        if self._task_type == 'classification':
            return GPTClassificationHead(deepcopy(self.config))
        elif self._task_type == 'language_model':
            layer = GPTLMHead(deepcopy(self.config))
            if self.load_pretrained_model:
                layer.load_state_dict(
                    torch.load(f'{self.model_name}/pytorch_lm_head.pt')
                )
            return layer
        raise Exception('unknown data type')

    def _create_transformer_layer(self, layer_idx=0):
        config = deepcopy(self.config)
        layer = GPTBlock(config, layer_id=layer_idx) # TODO: checkpoint
        if self.load_pretrained_model:
            print(f'loading layer {layer_idx}')
            layer.load_state_dict(
                torch.load(f'{self.model_name}/pytorch_{layer_idx}.pt')
            )
        return layer


class GPTStageFirst(GPTStageBase):
    def __init__(self, args, config, device):
        super(GPTStageFirst, self).__init__(args, config)
        self.device = device
        module_list = [self._create_first_layer()]
        for layer_idx in range(self._layer_begin, self._layer_end):
            module_list.append(self._create_transformer_layer(layer_idx=layer_idx))
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device))
        return out.cpu() if self._to_cpu else out


class GPTStageMiddle(GPTStageBase):
    def __init__(self, args, config, device):
        super(GPTStageMiddle, self).__init__(args, config)
        self.device = device
        module_list = []
        for layer_idx in range(self._layer_begin, self._layer_end):
            module_list.append(self._create_transformer_layer(layer_idx=layer_idx))
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        return out.cpu() if self._to_cpu else out


class GPTStageLast(GPTStageBase):
    def __init__(self, args, config, device):
        super(GPTStageLast, self).__init__(args, config)
        self.device = device
        module_list = []
        for layer_idx in range(self._layer_begin, self._layer_end):
            module_list.append(self._create_transformer_layer(layer_idx=layer_idx))
        module_list.append(self._create_last_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x, input_ids=None):
        if input_ids is None:
            out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        else:
            x = x.to(self.device) if self._to_cpu else x
            input_ids = input_ids.to(self.device) if self._to_cpu else input_ids
            for layer in self.model[:-1]:
                x = layer(x)
            out = self.model[-1](x, input_ids=input_ids)
            
        return out.cpu() if self._to_cpu else out