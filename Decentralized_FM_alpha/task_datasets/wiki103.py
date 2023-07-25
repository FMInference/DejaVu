import os
import re
import torch
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset, load_from_disk


def wikitext_detokenize(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string

    

    
def get_wiki103_train_data_loader(args, tokenizer, num_workers=0):
    
    if os.path.isdir('./data/wiki103_train_ready'):
        train_set = load_from_disk('./data/wiki103_train_ready')
    else:
        data = load_from_disk("./data/wiki103/train")
        encodings = tokenizer("\n\n".join(
            [wikitext_detokenize(t) for t in data["text"]]
        ), return_tensors="pt")

        input_ids_list = []
        stride = args.seq_length
        for i in tqdm(range(0, encodings.input_ids.size(1)-stride, stride)):
            begin_loc = i
            end_loc = min(i+stride, encodings.input_ids.size(1))
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            input_ids_list.append(input_ids)
        input_ids = torch.cat(input_ids_list, 0)

        train_set = Dataset.from_dict({
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids),
            'idx': list(range(len(input_ids))),
        })
        
        train_set.save_to_disk('./data/wiki103_train_ready')
    
    train_set = train_set.map(lambda examples: {'text': examples['input_ids']}, batched=True)
    train_set.set_format(
        type='torch', columns=[
            'text', 'input_ids', 'attention_mask', 'idx',
        ])
    
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_sampler = torch.utils.data.RandomSampler(train_set, generator=generator)
    train_data_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=args.batch_size,
                                                    sampler=train_sampler,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader
    
    
def get_wiki103_test_data_loader(args, tokenizer, num_workers=0):
    
    data = load_from_disk("./data/wiki103/test")
    encodings = tokenizer("\n\n".join(
        [wikitext_detokenize(t) for t in data["text"]]
    ), return_tensors="pt")
    
    input_ids_list = []
#     window = args.seq_length # TODO: a smaller value
#     for i in range(window, encodings.input_ids.size(1)):
#         begin_loc = max(i - window, 0)
#         end_loc = min(i, encodings.input_ids.size(1))
#         input_ids = encodings.input_ids[:, begin_loc:end_loc]
#         input_ids_list.append(input_ids)
#     input_ids = torch.cat(input_ids_list, 0)
    stride = args.seq_length
    # TODO: last stride is dropped
    for i in tqdm(range(0, encodings.input_ids.size(1)-stride, stride)):
        begin_loc = i
        end_loc = min(i+stride, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        input_ids_list.append(input_ids)
    input_ids = torch.cat(input_ids_list, 0)
    
    train_set = Dataset.from_dict({
        'input_ids': input_ids,
        'attention_mask': torch.ones_like(input_ids),
        'idx': list(range(len(input_ids))),
    })
    
    train_set = train_set.map(lambda examples: {'text': examples['input_ids']}, batched=True)
    train_set.set_format(
        type='torch', columns=[
            'text', 'input_ids', 'attention_mask', 'idx',
        ])
    
    # TODO: let drop_last be False
    train_data_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    collate_fn=None)
        
    return train_data_loader