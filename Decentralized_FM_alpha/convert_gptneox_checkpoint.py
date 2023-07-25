import os
import argparse
import torch

if __name__ == '__main__':
    
    try:
        os.mkdir('gpt-neox-20b-new')
    except:
        pass

    with open('gpt-neox-20b/pytorch_model.bin.index.json') as f:
        index = json.load(f)

    ## emb
    item = {}
    item['embed_in.weight'] = torch.load(
        'gpt-neox-20b/' + index['weight_map']['gpt_neox.embed_in.weight'],
        map_location=torch.device('cpu'),
    )['gpt_neox.embed_in.weight']
    torch.save(item, 'gpt-neox-20b-new/pytorch_embs.pt')


    ## out
    item = {}
    item['embed_out.weight'] = torch.load(
        'gpt-neox-20b/' + index['weight_map']['embed_out.weight'],
        map_location=torch.device('cpu'),
    )['embed_out.weight']

    item['final_layer_norm.weight'] = torch.load(
        'gpt-neox-20b/' + index['weight_map']['gpt_neox.final_layer_norm.weight'],
        map_location=torch.device('cpu'),
    )['gpt_neox.final_layer_norm.weight']

    item['final_layer_norm.bias'] = torch.load(
        'gpt-neox-20b/' + index['weight_map']['gpt_neox.final_layer_norm.bias'],
        map_location=torch.device('cpu'),
    )['gpt_neox.final_layer_norm.bias']

    torch.save(item, 'gpt-neox-20b-new/pytorch_lm_head.pt')
    
    
    ## layers

    for i in range(0, 44):
        layer_prefix = f'gpt_neox.layers.{i}.'

        item = {}

        layer_maps = {k:v for k,v in index['weight_map'].items() if k.startswith(layer_prefix)}

        caches = {}

        for k, v in layer_maps.items():
            new_k = k.replace(layer_prefix, '')
            to_read = 'gpt-neox-20b/' + index['weight_map'][k]
            if to_read not in caches:
                caches[to_read] = torch.load(to_read,map_location=torch.device('cpu'))
            item[new_k] = caches[to_read][k]

        torch.save(item, f'gpt-neox-20b-new/pytorch_{i}.pt')

        del item
        del caches
