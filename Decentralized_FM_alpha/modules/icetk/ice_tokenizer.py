# -*- encoding: utf-8 -*-

import os
import sys
import math
import random
from typing import List, Tuple, Union

import torch
# from PIL import Image
# from torchvision import transforms
# from torchvision.transforms.functional import pil_to_tensor

from .text_tokenizer import TextTokenizer
# from .image_tokenizer import ImageTokenizer
from .utils import auto_create

class IceTokenizer:
    def __init__(self, path='~/.icetk_models', device='cuda', fp16=True):
        self.configure(path, device, fp16)
        
    def configure(self, path=None, device=None, fp16=None):
        if path is not None:
            self.path = os.path.expanduser(path)
        if device is not None:
            self.device = device
        if fp16 is not None:
            self.fp16 = fp16
            
    @property
    def text_tokenizer(self):
        if not hasattr(self, '_text_tokenizer'):
            fp = os.path.join(self.path, 'ice_text.model')
            auto_create(fp)
            self._text_tokenizer = TextTokenizer(fp)
        return self._text_tokenizer
    
    # @property
    # def image_tokenizer(self):
    #     if not hasattr(self, '_image_tokenizer'):
    #         fp = os.path.join(self.path, 'ice_image.pt')
    #         auto_create(fp)
    #         self._image_tokenizer = ImageTokenizer(fp, device=self.device, fp16=self.fp16)
    #     return self._image_tokenizer
    
    @property
    def num_image_tokens(self):
        return 20000 # self.image_tokenizer.num_tokens # allow not load
    
    @property
    def num_text_tokens(self):
        return self.text_tokenizer.num_tokens
    @property
    def num_tokens(self):
        return self.num_image_tokens + self.num_text_tokens
        
    def add_special_tokens(self, special_tokens: List[str]):
        self.text_tokenizer.add_special_tokens(special_tokens)
    
    def encode(self, text=None, 
               image_path=None, image_pil=None, image_torch=None, 
               image_size: int=None, compress_rate=8, ignore_linebreak=True):
        assert (text is None) + (image_path is None) + (image_pil is None) + (image_torch is None) == 3
        assert int(compress_rate) in [4, 8, 16]
        if text is not None:
            if not ignore_linebreak:
                text = text.replace('\n', '<n>')
            tmp = self.text_tokenizer.encode(text)
            return [x + self.num_image_tokens for x in tmp]
        else:
            raise Exception('img tokenizer is missing')
            # need_norm_to_1 = False
            # if image_path is not None:
            #     image_pil = Image.open(image_path)
            # if image_torch is None:
            #     image_torch = pil_to_tensor(image_pil)
            #     need_norm_to_1 = True
            # if image_size is not None:
            #     # for speed in large-scale preprocessing, set this to None and transform in Dataloader.
            #     # TODO: test speed
            #     tr = transforms.Compose([
            #         transforms.Resize(image_size),
            #         transforms.CenterCrop(image_size),
            #     ])
            #     image_torch = tr(image_torch)
            # image_torch = image_torch.to(self.image_tokenizer.device).float()
            # if need_norm_to_1:
            #     image_torch /= 255.
            # return self.image_tokenizer.encode(image_torch, l=int(math.log2(compress_rate))-2)
            

    def decode(self, text_ids: List[int]=None, image_ids: Union[List[int], torch.LongTensor]=None, compress_rate=8):
        assert (text_ids is None) + (image_ids is None) == 1
        if text_ids is not None:
            ids = [int(_id) - self.num_image_tokens for _id in text_ids]
            if any([i < 0 or i >= self.num_text_tokens for i in ids]):
                print('warning:', ids)
                print(f'should between 0 and {self.num_text_tokens}')
                ids = [0 if i < 0 or i >= self.num_text_tokens else i for i in ids]
            return self.text_tokenizer.decode(ids).replace('<n>', '\n')
        else:
            raise Exception('img tokenizer is missing')
            # return self.image_tokenizer.decode(image_ids, l=int(math.log2(compress_rate))-2)
            
    def tokenize(self, text):
        return self.text_tokenizer.tokenize(text)

    def __getitem__(self, x):
        if isinstance(x, int):
            if x < self.num_image_tokens:
                return '<image_{}>'.format(x)
            else:
                return self.text_tokenizer.convert_id_to_token(x - self.num_image_tokens)
        elif isinstance(x, str):
            if x.startswith('<image_') and x.endswith('>') and x[7:-1].isdigit():
                return int(x[7:-1])
            else:
                return self.text_tokenizer.convert_token_to_id(x) + self.num_image_tokens
        else:
            raise ValueError('The key should be str or int.')
        
        
