import argparse
import random
import os
import torch.autograd.profiler as profiler
from task_datasets.wikitext import get_wikitext_train_data_loader, get_wikitext_test_data_loader
from task_datasets.wiki103 import get_wiki103_train_data_loader, get_wiki103_test_data_loader
from task_datasets.arxiv21 import get_arxiv21_train_data_loader, get_arxiv21_test_data_loader
from task_datasets.openwebtext import get_openwebtext_train_data_loader
from task_datasets.fm_in_context_eval_data import get_fm_in_context_eval_train_data_loader
from pipeline_parallel.dist_pp_utils import get_pp_finetune_module as get_pp_module
from coordinator.lsf.lsf_coordinate_client_deprecated import CoordinatorTrainClient
from transformers import AutoTokenizer, AutoConfig

try:
    import wandb
except Exception as e:
    wandb = None
from utils.dist_args_utils import *
from utils.dist_train_utils import *
from utils.dist_test_utils import *
from comm.comm_utils import *


def save_checkpoint(args, pipe, ckpt_path):
    
    pp_rank = get_pipeline_parallel_rank()
    _layer_begin = pp_rank * args.num_layers
    _layer_end = min(_layer_begin + args.num_layers, args.max_layers)
    
    # if hasattr(pipe)
    torch.save(
        pipe.scheduler.state_dict(),
        os.path.join(ckpt_path, f'scheduler_rank_{pp_rank}.pt')
    )
    
    torch.save(
        pipe.optimizer.state_dict(),
        os.path.join(ckpt_path, f'optimizer_rank_{pp_rank}.pt')
    )
    
    if pp_rank  == 0:
        torch.save(
            pipe.model.model[0].state_dict(),
            os.path.join(ckpt_path, f'pytorch_embs.pt')
        )
        
        for i in range(_layer_begin, _layer_end):
            print('saving layer', i)
            torch.save(
                pipe.model.model[i+1-_layer_begin].state_dict(),
                os.path.join(ckpt_path, f'pytorch_{i}.pt')
            )
            
    elif pp_rank  == args.pipeline_group_size - 1:
        for i in range(_layer_begin, _layer_end):
            print('saving layer', i)
            torch.save(
                pipe.model.model[i-_layer_begin].state_dict(),
                os.path.join(ckpt_path, f'pytorch_{i}.pt')
            )
        torch.save(
            pipe.model.model[-1].state_dict(),
            os.path.join(ckpt_path, f'pytorch_lm_head.pt')
        )
    else:
        for i in range(_layer_begin, _layer_end):
            print('saving layer', i)
            torch.save(
                pipe.model.model[i-_layer_begin].state_dict(),
                os.path.join(ckpt_path, f'pytorch_{i}.pt')
            )

def train_loop(args, pipe, device, train_data_loader, test_data_loader):
    
    pp_rank = get_pipeline_parallel_rank()
    
    if os.path.isfile(os.path.join(args.model_name, f'scheduler_rank_{pp_rank}.pt')):
        print('resuming scheduler')
        pipe.scheduler.load_state_dict(
            torch.load(
                os.path.join(args.model_name, f'scheduler_rank_{pp_rank}.pt')
            )
        )
        
    if os.path.isfile(os.path.join(args.model_name, f'optimizer_rank_{pp_rank}.pt')):
        print('resuming optimizer')
        pipe.optimizer.load_state_dict(
            torch.load(
                os.path.join(args.model_name, f'optimizer_rank_{pp_rank}.pt')
            )
        )
    
    for e in range(args.n_epochs):
        
        distributed_train_lm_iter(args, pipe, device, train_data_loader)
        
        if test_data_loader is not None and args.do_evaluation:
            distributed_test_lm_iter(args, pipe, device, test_data_loader)
            
        if get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
            if wandb is not None:
                wandb.log({'epoch': e}, step=pipe.global_step)
                
        if args.model_save_path is not None:
            ckpt_path = os.path.join(args.model_save_path, f'ckpt_step_{pipe.global_step}')
            try:
                os.makedirs(ckpt_path)
            except Exception as e:
                pass
                
            save_checkpoint(args, pipe, ckpt_path)
            

def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    add_torch_distributed_w_euler_coordinator_arguments(parser)
    add_device_arguments(parser)
    # add_torch_distributed_arguments(parser)
    add_training_model_arguments(parser)
    # add_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    add_mixed_precision_arguments(parser)
    add_parallel_schema_arguments(parser)
    add_finetuning_model_arguments(parser)
    parser.add_argument('--profiling', type=str, default='tidy_profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
    parser.add_argument('--trace-postfix', type=str, default='default', metavar='S',
                        help='postfix of the tracing file name.')
    parser.add_argument('--do-evaluation', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='do evaluation or not.')
    args = parser.parse_args()
    
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')

    coord_client = CoordinatorTrainClient(args)
    prime_ip, rank = coord_client.notify_train_join()
    print("<====Coordinator assigned prime-IP:", prime_ip, " and my assigned rank", rank, "====>")

    args.rank = rank
    init_communicators_with_coordinator(args, prime_ip, rank)
    
    config = AutoConfig.from_pretrained(args.model_name)
    for k in config.__dict__:
        if '_pdrop' in k or '_dropout' in k:
            config.__dict__[k] = args.dropout
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.model_max_length = args.seq_length
    config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    print("token vocab size:", tokenizer.vocab_size)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if get_pipeline_parallel_rank() == 0:
        if args.task_name == 'wikitext':
            train_data_loader = get_wikitext_train_data_loader(args, tokenizer)
            test_data_loader = get_wikitext_test_data_loader(args, tokenizer)
        elif args.task_name == 'wiki103':
            train_data_loader = get_wiki103_train_data_loader(args, tokenizer)
            test_data_loader = get_wiki103_test_data_loader(args, tokenizer)
        elif args.task_name == 'arxiv21':
            train_data_loader = get_arxiv21_train_data_loader(args, tokenizer)
            test_data_loader = get_arxiv21_test_data_loader(args, tokenizer)
        elif args.task_name == 'openwebtext':
            train_data_loader = get_openwebtext_train_data_loader(args, tokenizer)
            test_data_loader = get_wikitext_test_data_loader(args, tokenizer)
        elif args.task_name == 'fm_in_context_eval':
            train_data_loader = get_fm_in_context_eval_train_data_loader(args, tokenizer)
            test_data_loader = None
        else:
            raise Exception('unknown task.')
    else:
        train_data_loader, test_data_loader = None, None
            
    if args.warmup_steps is None:
        args.warmup_steps = 0 #len(train_data_loader)
    if args.total_steps is None:
        args.total_steps = 0 #len(train_data_loader) * args.n_epochs
    
    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        print("Running ", args.pp_mode, " with data parallel.")
    else:
        print("Running ", args.pp_mode, " without data parallel.")

#     torch.manual_seed(args.seed)
#     random.seed(args.seed)
#     np.random.seed(args.seed)
    
    print('initializing pipeline')
    pipe = get_pp_module(args, config, device, use_dp)

    print('starting train loop....')
    if args.profiling == 'no-profiling':
        train_loop(args, pipe, device, train_data_loader, test_data_loader)
    else:
        prefix = './trace_json/gpt3_' + args.pp_mode
        if use_dp:
            prefix = prefix + '_' + args.dp_mode
        trace_file = prefix + get_learning_arguments_str(args) + get_model_arguments_str(args) + \
                     get_dist_arguments_str(args) + get_mixed_precision_arguments_str(args) + '_' + \
                     args.profiling + '_' + args.trace_postfix + '.json'
        if args.profiling == 'tidy_profiling':
            try:
                train_loop(args, pipe, device, train_data_loader, test_data_loader)
            except Exception as e:
                raise e
                print(get_pipeline_parallel_rank(), e)
            pipe.export_profiling_result(filename=trace_file)
        elif args.profiling == 'pytorch_profiling':
            with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
                train_loop(args, pipe, device, train_data_loader, test_data_loader)
            print(prof.key_averages().table())
            prof.export_chrome_trace(trace_file)
        else:
            print("No recognized profiler?")
            assert False
    print(get_pipeline_parallel_rank(), 'finished.')
    train_finish_msg = str(rank) + '#' + str(round(0.0, 3))
    coord_client.notify_train_finish(message=train_finish_msg)

if __name__ == '__main__':

    import datasets
    # euler distributed file system makes cache slow
    datasets.disable_caching()

    main()
