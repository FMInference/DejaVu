import os
import argparse
import torch

if __name__ == '__main__':
    
    try:
        os.mkdir('t5-11b-new')
    except:
        pass

    with open('t5-11b/pytorch_model.bin.index.json') as f:
        index = json.load(f)

    ## emb
    item = {}
    item['shared.weight'] = torch.load(
        't5-11b/' + index['weight_map']['shared.weight'],
        map_location=torch.device('cpu'),
    )['shared.weight']
    torch.save(item, 't5-11b-new/pytorch_embs.pt')
    
    
    ## out
    item = {}

    item['final_layer_norm.weight'] = torch.load(
        't5-11b/' + index['weight_map']['encoder.final_layer_norm.weight'],
        map_location=torch.device('cpu'),
    )['encoder.final_layer_norm.weight']

    torch.save(item, 't5-11b-new/pytorch_enc_head.pt')
    
    
    ## out
    item = {}
    item['lm_head.weight'] = torch.load(
        't5-11b/' + index['weight_map']['lm_head.weight'],
        map_location=torch.device('cpu'),
    )['lm_head.weight']

    item['final_layer_norm.weight'] = torch.load(
        't5-11b/' + index['weight_map']['decoder.final_layer_norm.weight'],
        map_location=torch.device('cpu'),
    )['decoder.final_layer_norm.weight']

    torch.save(item, 't5-11b-new/pytorch_dec_head.pt')


    ## layers

    for i in range(0, 24):
        layer_prefix = f'encoder.block.{i}.'

        item = {}

        layer_maps = {k:v for k,v in index['weight_map'].items() if k.startswith(layer_prefix)}
        layer_maps['layer.0.SelfAttention.relative_attention_bias.weight'] = index[
            'weight_map']['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']

        for k, v in layer_maps.items():
            new_k = k.replace(layer_prefix, '')
            item[new_k] = torch.load(
                't5-11b/' + index['weight_map'][k],
                map_location=torch.device('cpu'),
            )[k]

        torch.save(item, f't5-11b-new/pytorch_enc_{i}.pt')

        del item

        del item
        del caches
        
    for i in range(0, 24):
        layer_prefix = f'decoder.block.{i}.'

        item = {}

        layer_maps = {k:v for k,v in index['weight_map'].items() if k.startswith(layer_prefix)}
        layer_maps['layer.0.SelfAttention.relative_attention_bias.weight'] = index[
            'weight_map']['decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']

        for k, v in layer_maps.items():
            new_k = k.replace(layer_prefix, '')
            item[new_k] = torch.load(
                't5-11b/' + index['weight_map'][k],
                map_location=torch.device('cpu'),
            )[k]

        torch.save(item, f't5-11b-new/pytorch_dec_{i}.pt')

        del item
