import os
import argparse
import torch

if __name__ == '__main__':
    
    try:
        os.mkdir('opt-66b-new')
    except:
        pass

    with open('opt-66b/pytorch_model.bin.index.json') as f:
        index = json.load(f)

    ## emb
    item = {}
    item['embed_tokens.weight'] = torch.load(
        'opt-66b/' + index['weight_map']['model.decoder.embed_tokens.weight'],
        map_location=torch.device('cpu'),
    )['model.decoder.embed_tokens.weight']
    item['embed_positions.weight'] = torch.load(
        'opt-66b/' + index['weight_map']['model.decoder.embed_positions.weight'],
        map_location=torch.device('cpu'),
    )['model.decoder.embed_positions.weight']
    torch.save(item, 'opt-66b-new/pytorch_embs.pt')


    ## out
    item = {}
    item['lm_head.weight'] = torch.load(
        'opt-66b/' + index['weight_map']['model.decoder.embed_tokens.weight'],
        map_location=torch.device('cpu'),
    )['model.decoder.embed_tokens.weight']

    item['final_layer_norm.weight'] = torch.load(
        'opt-66b/' + index['weight_map']['model.decoder.final_layer_norm.weight'],
        map_location=torch.device('cpu'),
    )['model.decoder.final_layer_norm.weight']

    item['final_layer_norm.bias'] = torch.load(
        'opt-66b/' + index['weight_map']['model.decoder.final_layer_norm.bias'],
        map_location=torch.device('cpu'),
    )['model.decoder.final_layer_norm.bias']

    torch.save(item, 'opt-66b-new/pytorch_lm_head.pt')

    ## layers

    for i in range(0, 64):
        layer_prefix = f'model.decoder.layers.{i}.'
  
        item = {}
  
        layer_maps = {k:v for k,v in index['weight_map'].items() if k.startswith(layer_prefix)}
  
        for k, v in layer_maps.items():
            new_k = k.replace(layer_prefix, '')
            item[new_k] = torch.load(
                'opt-66b/' + index['weight_map'][k],
                map_location=torch.device('cpu'),
            )[k]
  
        torch.save(item, f'opt-66b-new/pytorch_{i}.pt')
  
        del item
