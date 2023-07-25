import torch
from pathlib import Path
from datasets import load_from_disk
from comm.comm_utils import *

from .fm_in_context_eval_utils import train_data_utils


def get_fm_in_context_eval_train_data_loader(args, tokenizer, num_workers=0):
    
    block_size = args.seq_length
    
    data_cache_dir = (
        Path(args.data_cache_dir)
        / "_".join(Path(d).name for d in args.data_dirs)
        / f"{int(1)}mask_{int(-1)}maxev_{int(-1)}maxtr_{-1}maxtrpr_{int(False)}cb_{block_size}bs"
    )
    
    print('loading raw data')
    hf_datasets_with_label, hf_datasets_without_label = train_data_utils.read_data(
        data_dirs=args.data_dirs,
        class_balanced=False,
        max_train_samples=-1,
        max_train_percent=-1,
        local_rank=0,
    )
    
    # Trick to get HF to cache the tokenization - must save/load from disk
    if args.rank in [0, -1]:
        hf_datasets_with_label.save_to_disk(
            data_cache_dir / "hf_datasets_with_label.dataset"
        )
        hf_datasets_without_label.save_to_disk(
            data_cache_dir / "hf_datasets_without_label.dataset"
        )
        
    # if torch.distributed.is_available() and torch.distributed.is_initialized():
    #     torch.distributed.barrier()
    '''
    comm = get_pipeline_parallel_comm()
    comm.barrier()
    if args.data_group_size > 1:
        comm = get_data_parallel_comm()
        comm.barrier()
    '''

    print('loading cached data')
    hf_datasets_with_label = load_from_disk(
        data_cache_dir / "hf_datasets_with_label.dataset"
    )
    hf_datasets_without_label = load_from_disk(
        data_cache_dir / "hf_datasets_without_label.dataset"
    )
    
    try:
        hf_datasets_with_label.pop('validation')
    except:
        pass
    try:
        hf_datasets_without_label.pop('validation')
    except:
        pass
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    class DummyContext:
        def __init__(self, *args, **kargs):
            pass
        def __enter__(self, *args, **kargs):
            pass
        def __exit__(self, *args, **kargs):
            pass

    class DummyArgs:
        def __init__(self):
            self.main_process_first = DummyContext
            self.local_rank = args.rank

    dummy_args = DummyArgs()
    
    print('tokenizing data')
    if len(hf_datasets_with_label) > 0:
        lm_datasets_with_label = train_data_utils.tokenize_data(
            hf_datasets=hf_datasets_with_label,
            tokenizer=tokenizer,
            preprocessing_num_workers=32,
            overwrite_cache=False,
            ignore_label_column=False,
            data_cache_dir=data_cache_dir,
            block_size=block_size,
            mask_input=1,
            training_args=dummy_args,
        )
    else:
        lm_datasets_with_label = {}

    if len(hf_datasets_without_label) > 0:
        lm_datasets_without_label = train_data_utils.tokenize_data(
            hf_datasets=hf_datasets_without_label,
            tokenizer=tokenizer,
            preprocessing_num_workers=32,
            overwrite_cache=False,
            ignore_label_column=True,
            data_cache_dir=data_cache_dir,
            block_size=1024,
            mask_input=1,
            training_args=dummy_args,
        )
    else:
        lm_datasets_without_label = {}
        
    print('merging data')
    lm_datasets = train_data_utils.merge_datasets(
        lm_datasets_with_label, lm_datasets_without_label
    )
    
    train_set = lm_datasets['train']

    # train_set = train_set.map(lambda examples: {'text': examples['input_ids']}, batched=True)
    
    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        dp_rank = get_data_parallel_rank()
    else:
        dp_rank = 0
    
    print('data loading')
    generator = torch.Generator()
    generator.manual_seed(args.seed + dp_rank)
    train_sampler = torch.utils.data.RandomSampler(train_set, generator=generator)

    def collate_fn(examples):
        keys = examples[0].keys()
        ret = {}
        for k in keys:
            ret[k] = torch.tensor([e[k] for e in examples])
        return ret

    train_data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn)
    
    print('returning dataloader')

    return train_data_loader
