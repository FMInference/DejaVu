
import torch
    
class GLMTokenizer:
    """Hardcoded tokenizer."""

    def __init__(self):
        
        self.tokenizer = None
        
        from .icetk import icetk
        self.tokenizer = icetk
            
        self.num_tokens = 150000
        self.add_special_tokens(['[MASK]', '[gMASK]', '[sMASK]', 'eod', 'sop', 'eop', 'ENC', 'dBLOCK'])
        self.sentence_end_decoder = {20007: '.', 20031: '？', 20035: '！', 20027: '；', 20012: ':', 83823: '。', 145670: '…'}

        self.special_tokens['eos'] = 20002
        self.special_tokens_decoder[20002] = '</s>'
        
        self.bos_token = "sop"
        self.eos_token = "eop"
        self.pad_token = "[pad]"
        
        self.model_max_length = 2048
        self.truncation_side = 'left'
        self.padding_side = 'left'
        
        self.echo_prompt = False

    def add_special_tokens(self, special_tokens):
        """Add a list of additional tokens to the encoder.
        The additional tokens are indexed starting from the last index of the
        current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, self.num_tokens + i) for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens.items()}
        # for k, v in self.special_tokens.items():
        #     self.tokenizer.decoder[v] = "\u0120" + k
        # logger.info("Special tokens {}".format(self.special_tokens))

    def get_command(self, token):
        return self.special_tokens[token]

    def contains_sentence_end(self, idx):
        return idx in self.sentence_end_decoder

    def IdToToken(self, idx):
        if idx == 0:
            return '[pad]'
        elif idx in self.special_tokens_decoder:
            return f"[{self.special_tokens_decoder[idx]}]"
        else:
            return self.tokenizer.decode([idx])

    def TokenToId(self, token):
        if token == '[pad]':
            return 0
        elif token in self.special_tokens:
            return self.special_tokens[token]
        else:
            return self.tokenizer.encode(token)[0]
        
    @property
    def vocab_size(self):
        return self.num_tokens + len(self.special_tokens)

    @property
    def vocab(self):
        assert False
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        assert False
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        split = [-1]
        for i, token in enumerate(token_ids):
            if token in self.special_tokens_decoder:
                split.append(i)
        split.append(len(token_ids))
        text = ""
        for i in range(len(split) - 1):
            if i > 0:
                text += self.IdToToken(token_ids[split[i]])
            text += self.tokenizer.decode(token_ids[split[i] + 1: split[i + 1]])
        return text

    @property
    def eod(self):
        return self.get_special_token('eod')
        
    @classmethod
    def from_pretrained(cls, model_name_or_path=None):
        return cls()
    
    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.TokenToId(tokens)
        return [self.TokenToId(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.IdToToken(ids)
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().tolist()
        return [self.IdToToken(idx) for idx in ids]
    
    def __call__(self, text, return_tensors='pt', padding='max_length', truncation=True, add_gmask=True):
        
        assert return_tensors == 'pt'
        assert padding == 'max_length' or padding == True
        
        if isinstance(text, str):
            text = [text]
            
        ids = []
        for t in text:
            t_ids = self.tokenize(t)
            if add_gmask:
                if not self.echo_prompt:
                    t_ids = t_ids + [self.get_command('[gMASK]'), self.get_command('sop')] # append <s>
                else:
                    t_ids = [self.get_command('[gMASK]'), self.get_command('sop')] + t_ids
            
            if truncation:
                if self.truncation_side == 'left':
                    t_ids = t_ids[-self.model_max_length:]
                else:
                    t_ids = t_ids[:self.model_max_length]
                    
            assert self.get_command('[gMASK]') in t_ids
            
            ids.append(t_ids)
        
        if padding != 'max_length':
            max_len = max([len(t_ids) for t_ids in ids])
        else:
            max_len = self.model_max_length
        
        attention_mask = torch.ones(len(ids), max_len, dtype=torch.long)
        
        if self.padding_side == 'left':
            new_ids = []
            for i, t_ids in enumerate(ids):
                attention_mask[i, :max_len - len(t_ids)] = 0
                new_ids.append([0]*(max_len - len(t_ids)) + t_ids) # 0 is [pad]
        else:
            new_ids = []
            for i, t_ids in enumerate(ids):
                attention_mask[i, -(max_len - len(t_ids)):] = 0
                new_ids.append(t_ids + [0]*(max_len - len(t_ids))) # 0 is [pad]
        ids = new_ids
        ids = torch.tensor(ids)
        
        return {
            'input_ids': ids, 'attention_mask': attention_mask
        }
        
    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        return self.detokenize(token_ids)
        