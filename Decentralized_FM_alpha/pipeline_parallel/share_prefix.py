import time
import json
import torch
import torch.nn.functional as F

class Trie:

    def __init__(self):
        self.__final = False
        self.__value = None
        self.__depth = 0
        self.__nodes = {}
        
    def value(self):
        return self.__value
        
    def depth(self):
        return self.__depth
        
    def nodes(self):
        return self.__nodes
    
    def match(self, array):
        ret = self
        for x in array:
            if x in ret.__nodes:
                ret = ret[[x]]
            else:
                break
        return ret

    def __repr__(self):
        return 'Trie<len={}, final={}>'.format(len(self), self.__final)

    def __getstate__(self):
        return self.__final, self.__nodes

    def __setstate__(self, state):
        self.__final, self.__nodes = state

    def __len__(self):
        return len(self.__nodes)

    def __bool__(self):
        return self.__final

    def __contains__(self, array):
        try:
            return self[array]
        except KeyError:
            return False

    def __iter__(self):
        yield self
        for node in self.__nodes.values():
            yield from node

    def __getitem__(self, array):
        return self.__get(array, False)

    def create(self, array, value=None):
        ret = self.__get(array, True, value)
        ret.__final = True

    def read(self):
        yield from self.__read([])

    def update(self, array, value=None):
        ret = self[array]
        ret.__final = True
        if value is not None:
            ret.__value = value

    def delete(self, array):
        self[array].__final = False

    def prune(self):
        for key, value in tuple(self.__nodes.items()):
            if not value.prune():
                del self.__nodes[key]
        if not len(self):
            self.delete([])
        return self

    def __get(self, array, create, value=None):
        if array is not None and len(array) > 0:
            head, *tail = array
            if create and head not in self.__nodes:
                new_t = Trie()
                new_t.__value = value
                new_t.__depth = self.__depth + 1
                self.__nodes[head] = new_t
            return self.__nodes[head].__get(tail, create, value=value)
        return self

    def __read(self, name):
        if self.__final:
            yield name
        for key, value in self.__nodes.items():
            yield from value.__read(name + [key])
            
            
class SharePrefix:
    def __init__(self):
        self.enabled = True
        self.clear()
        
    def enable(self, enabled=True):
        self.enabled = enabled
    
    def disable(self):
        self.enabled = False
    
    def clear(self):
        self.trie = Trie()
        
    def insert(self, input_ids, attention_mask, k_v_caches):
        input_ids = input_ids.cpu().numpy()
        begin = attention_mask.argmax(-1).item()
        self.trie.create(
            input_ids[begin:], 
            [
                (k[:,:,begin:], v[:,:,begin:]) for k, v in k_v_caches
            ]
        )
    
    def search(self, input_ids, attention_mask):
        input_ids = input_ids.cpu().numpy()
        begin = attention_mask.argmax(-1).item()
        ret_node = self.trie.match(input_ids[begin:])
        if ret_node.depth() > 0:
            end = begin + ret_node.depth()
            # avoid zero dimension
            if end >= len(input_ids):
                end = len(input_ids) - 1
            ret_k_v_caches = []
            for ret_k, ret_v in ret_node.value():
                ret_k = ret_k[:,:,:end-begin]
                ret_v = ret_v[:,:,:end-begin]
                ret_k = F.pad(ret_k, (0,0,begin,0), "constant", 0.0)
                ret_v = F.pad(ret_v, (0,0,begin,0), "constant", 0.0)
                ret_k_v_caches.append((ret_k, ret_v))
            return ret_k_v_caches, begin, end
        else:
            return None, 0, 0
        
    def process_inputs(self, input_ids, attention_mask, input_embs, caches):
        
        if not self.enabled:
            return attention_mask, input_embs, caches
        
        past_caches, past_begin, past_end = self.search(input_ids.squeeze(0), attention_mask.squeeze(0))
        if past_caches is None:
            return attention_mask, input_embs, caches
        else:
            print(f'match prefix from {past_begin} to {past_end}!')
            return attention_mask, input_embs[:, past_end:], past_caches
        
    def process_outputs(self, input_ids, attention_mask, output_embs, caches):
        
        if not self.enabled:
            return output_embs
        
        diff = attention_mask.size(1) - output_embs.size(1) 
        if diff != 0:
            output_embs = F.pad(output_embs, (0,0,diff,0), "constant", 0.0)
        
        self.insert(input_ids.squeeze(0), attention_mask.squeeze(0), caches)
            
        return output_embs