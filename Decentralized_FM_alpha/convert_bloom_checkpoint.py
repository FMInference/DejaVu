import os
import argparse
import torch

if __name__ == '__main__':
    
    try:
        os.mkdir('bloom-new')
    except:
        pass

    with open('bloom/pytorch_model.bin.index.json') as f:
        index = json.load(f)

    ## emb
    item = {}
    item['word_embeddings.weight'] = torch.load(
        'bloom/' + index['weight_map']['word_embeddings.weight'],
        map_location=torch.device('cpu'),
    )['word_embeddings.weight']
    item['word_embeddings_layernorm.bias'] = torch.load(
        'bloom/' + index['weight_map']['word_embeddings_layernorm.bias'],
        map_location=torch.device('cpu'),
    )['word_embeddings_layernorm.bias']
    item['word_embeddings_layernorm.weight'] = torch.load(
        'bloom/' + index['weight_map']['word_embeddings_layernorm.weight'],
        map_location=torch.device('cpu'),
    )['word_embeddings_layernorm.weight']
    torch.save(item, 'bloom-new/pytorch_embs.pt')


    ## out
    item = {}
    item['lm_head.weight'] = torch.load(
        'bloom/' + index['weight_map']['word_embeddings.weight'],
        map_location=torch.device('cpu'),
    )['word_embeddings.weight']

    item['ln_f.weight'] = torch.load(
        'bloom/' + index['weight_map']['ln_f.weight'],
        map_location=torch.device('cpu'),
    )['ln_f.weight']

    item['ln_f.bias'] = torch.load(
        'bloom/' + index['weight_map']['ln_f.bias'],
        map_location=torch.device('cpu'),
    )['ln_f.bias']

    torch.save(item, 'bloom-new/pytorch_lm_head.pt')

    ## layers

    for i in range(0, 70):
        layer_prefix = f'h.{i}.'

        item = {}

        layer_maps = {k:v for k,v in index['weight_map'].items() if k.startswith(layer_prefix)}

        for k, v in layer_maps.items():
            new_k = k.replace(layer_prefix, '')
            item[new_k] = torch.load(
                'bloom/' + index['weight_map'][k],
                map_location=torch.device('cpu'),
            )[k]

        torch.save(item, f'bloom-new/pytorch_{i}.pt')

        del item
