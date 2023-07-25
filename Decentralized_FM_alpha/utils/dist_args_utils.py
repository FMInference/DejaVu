def add_device_arguments(parser):
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--cuda-num', type=int, default=1, metavar='N',
                        help='number of GPUs, if the instance has multiple GPUs.')
    parser.add_argument('--debug-mem', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, we will print some memory stats.')


def add_torch_distributed_arguments(parser):
    parser.add_argument('--dist-backend', type=str, default='cupy_nccl', metavar='S',
                        help='backend type for distributed PyTorch (default: cupy_nccl)')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9000', metavar='S',
                        help='master ip for distributed PyTorch')
    parser.add_argument('--world-size', type=int, default=4, metavar='D',
                        help='world-size (default: 4)')
    parser.add_argument('--pipeline-group-size', type=int, default=4, metavar='D',
                        help='world-size (default: 2)')
    parser.add_argument('--data-group-size', type=int, default=1, metavar='D',
                        help='world-size (default: 1)')
    parser.add_argument('--rank', type=int, default=0, metavar='N',
                        help='rank of the node')


def add_torch_distributed_w_euler_coordinator_arguments(parser):
    parser.add_argument('--dist-backend', type=str, default='cupy_nccl', metavar='S',
                        help='backend type for distributed PyTorch (default: cupy_nccl)')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--lsf-job-no', type=str, default='100', metavar='S',
                        help='Job-<ID> assigned by LSF.')
    parser.add_argument('--unique-port', type=str, default='100', metavar='S',
                        help='Which port to use, each client should have different value of this.')
    parser.add_argument('--world-size', type=int, default=4, metavar='D',
                        help='world-size (default: 4)')
    parser.add_argument('--pipeline-group-size', type=int, default=4, metavar='D',
                        help='world-size (default: 2)')
    parser.add_argument('--data-group-size', type=int, default=1, metavar='D',
                        help='world-size (default: 1)')
    parser.add_argument('--rank', type=int, default=0, metavar='N',
                        help='rank of the node')


def add_qqp_task_arguments(parser):
    parser.add_argument('--train-data', nargs='+', default=['./task_datasets/data/QQP/train.tsv'], metavar='S',
                        help='path to the training data')
    parser.add_argument('--valid-data', nargs='+', default=['./task_datasets/data/QQP/test.tsv'], metavar='S',
                        help='path to the training data')
    parser.add_argument('--tokenizer-type', type=str, default='BertWordPieceLowerCase', metavar='S',
                        help='which tokenizer to use.')
    parser.add_argument('--vocab-file', type=str, default='./task_datasets/data/bert-large-cased-vocab.txt', metavar='S',
                        help='which tokenizer to use.')
    parser.add_argument('--vocab-extra-ids', type=int, default=0, metavar='N',
                        help='-')
    parser.add_argument('--make-vocab-size-divisible-by', type=int, default=128, metavar='N',
                        help='-')


def add_training_model_arguments(parser):
    parser.add_argument('--seq-length', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--input-seq-length', type=int, default=16, metavar='N',
                        help='-')
    parser.add_argument('--generate-seq-length', type=int, default=16, metavar='N',
                        help='-')
    parser.add_argument('--embedding-dim', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--num-layers', type=int, default=2, metavar='N',
                        help='-')
    parser.add_argument('--num-heads', type=int, default=16, metavar='N',
                        help='-')
    parser.add_argument('--task', type=str, default='SeqClassification', metavar='S',
                        help='What task to run? SeqClassification or Seq2SeqClassification')


def add_finetuning_model_arguments(parser):
    parser.add_argument('--model-name', type=str, default='gpt2', metavar='S',
                        help='model name or path')
    parser.add_argument('--model-type', type=str, default='gpt2', metavar='S',
                        help='model type')
    parser.add_argument('--model-save-path', type=str, default=None, metavar='S',
                        help='model save path')
    parser.add_argument('--tokenizer-name', type=str, default='gpt2', metavar='S',
                        help='tokenizer name or path')
    parser.add_argument('--task-name', type=str, default='wikitext', metavar='S',
                        help='task name')
    parser.add_argument('--task-type', type=str, default='language_model', metavar='S',
                        help='task typw')
    parser.add_argument('--data-dirs', nargs='+', default=[],
                        help='data dirs for fm in context training')
    parser.add_argument('--data-cache-dir', type=str, default='', help='data cache dir for hf datasets tmp files / cache.')
    parser.add_argument('--n-epochs', type=int, default=10, help='-')
    # parser.add_argument('--warmup-epochs', type=int, default=1, help='warmup for activation compression')
    parser.add_argument('--warmup-steps', type=int, default=None, help='-')
    parser.add_argument('--total-steps', type=int, default=None, help='-')
    parser.add_argument('--load-pretrained-model', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='load pretrained model or not.')
    parser.add_argument('--max-layers', type=int, default=None, help='max layers')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')


def add_training_hyper_parameter_arguments(parser):
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--micro-batch-size', type=int, default=4, metavar='N',
                        help='input micro batch size for training (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                        help='-')
    parser.add_argument('--num-iters', type=int, default=4, metavar='N',
                        help='-')
    parser.add_argument('--weight-decay', type=float, default=0.1, metavar='N',
                        help='-')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='N',
                        help='-')


def add_mixed_precision_arguments(parser):
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode.')
    parser.add_argument('--loss-scale', type=float, default=64,
                        help='Static loss scaling, positive power of 2 values can improve fp16 convergence. ')
    parser.add_argument('--initial-loss-scale', type=float, default=2 ** 32,
                        help='Initial loss-scale for dynamic loss scaling.')
    parser.add_argument('--min-loss-scale', type=float, default=1.0,
                        help='Minimum loss scale for dynamic loss scale.')
    parser.add_argument('--loss-scale-window', type=float, default=1000,
                        help='Window over which to raise/lower dynamic scale.')
    parser.add_argument('--hysteresis', type=int, default=2,
                        help='hysteresis for dynamic loss scaling')
    parser.add_argument('--use-offload', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, we will offload the fp32 model to CPU RAM.')


def add_parallel_schema_arguments(parser):
    parser.add_argument('--pp-mode', type=str, default='gpipe', metavar='S',
                        help='use which pipeline parallel mode: gpipe or 1f1b.')
    parser.add_argument('--dp-mode', type=str, default='allreduce', metavar='S',
                        help='use which data parallel mode: allreduce.')
    parser.add_argument('--gradient-accumulate-step', type=int, default=1,
                        help='Number of gradient computation in Pipeline without data parallel sync.')


def get_model_arguments_str(args):
    return '_s' + str(args.seq_length) + '_m' + str(args.embedding_dim) + '_l' + str(args.num_layers)


def get_dist_arguments_str(args, add_rank=True, rank=None):
    dist_str = '_w' + str(args.world_size) + '_p' + str(args.pipeline_group_size) + "_" + \
               str(args.gradient_accumulate_step) + '_d' + str(args.data_group_size)
    if add_rank:
        if rank is not None:
            dist_str = dist_str + '_' + str(rank)
        else:
            dist_str = dist_str + '_' + str(args.rank)
    return dist_str


def get_learning_arguments_str(args):
    return '_b' + str(args.batch_size) + '_' + str(args.micro_batch_size)


def get_mixed_precision_arguments_str(args):
    arg_str = ''
    if args.fp16:
        arg_str = '_fp16'
    if args.use_offload:
        arg_str += '_offload'
    return arg_str


def add_torch_inference_distributed_arguments(parser):
    parser.add_argument('--dist-backend', type=str, default='cupy_nccl', metavar='S',
                        help='backend type for distributed PyTorch (default: cupy_nccl)')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9000', metavar='S',
                        help='master ip for distributed PyTorch')
    parser.add_argument('--pipeline-group-size', type=int, default=4, metavar='D',
                        help='world-size (default: 2)')
    parser.add_argument('--data-group-size', type=int, default=1, metavar='D',
                        help='world-size (default: 1)')
    parser.add_argument('--rank', type=int, default=0, metavar='N',
                        help='rank of the node')


def add_inference_arguments(parser):
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--micro-batch-size', type=int, default=1, metavar='N',
                        help='input micro batch size for training (default: 100)')
    parser.add_argument('--input-seq-length', type=int, default=16, metavar='N',
                        help='-')
    parser.add_argument('--generate-seq-length', type=int, default=16, metavar='N',
                        help='-')
    parser.add_argument('--num-layers', type=int, default=4, metavar='N',
                        help='-')
    parser.add_argument('--max-layers', type=int, default=None, metavar='N',
                        help='-')
    parser.add_argument('--pp-mode', type=str, default='pipe_greedy', metavar='S',
                        help='use which pipeline parallel mode: gpipe or 1f1b.')
    parser.add_argument('--num-iters', type=int, default=5, metavar='N',
                        help='-')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode.')


def add_hybrid_inference_arguments(parser):
    parser.add_argument('--node-type', type=str, default='CPU', metavar='S',
                        help='-')
    parser.add_argument('--producer-buffer-size', type=int, default=4, metavar='N',
                        help='-')
    parser.add_argument('--consumer-buffer-size', type=int, default=4, metavar='N',
                        help='-')
    parser.add_argument('--input-seq-length', type=int, default=16, metavar='N',
                        help='-')
    parser.add_argument('--generate-seq-length', type=int, default=16, metavar='N',
                        help='-')
    parser.add_argument('--stage-num-layers', type=int, default=4, metavar='N',
                        help='-')
    parser.add_argument('--global-num-layers', type=int, default=64, metavar='N',
                        help='-')
    parser.add_argument('--pp-mode', type=str, default='pipe_hybrid_greedy_async', metavar='S',
                        help='use which pipeline parallel mode: gpipe or 1f1b.')
    parser.add_argument('--num-iters', type=int, default=5, metavar='N',
                        help='-')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode.')


def add_torch_distributed_inference_w_euler_coordinator_arguments(parser):
    parser.add_argument('--dist-backend', type=str, default='cupy_nccl', metavar='S',
                        help='backend type for distributed PyTorch (default: cupy_nccl)')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--lsf-job-no', type=str, default='100', metavar='S',
                        help='Job-<ID> assigned by LSF.')
    parser.add_argument('--unique-port', type=str, default='100', metavar='S',
                        help='Which port to use, each client should have different value of this.')
    parser.add_argument('--pipeline-group-size', type=int, default=4, metavar='D',
                        help='world-size (default: 4)')
    parser.add_argument('--heartbeats-timelimit', type=float, default=60, metavar='S',
                        help='time to issue heartbeats')
    parser.add_argument('--working-directory', type=str,
                        default='/cluster/scratch/biyuan/fetch_cache', metavar='S',
                        help='The IP of coordinator-server.')


def add_torch_distributed_inference_w_crusoe_coordinator_arguments(parser):
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')


def add_lsf_coordinator_arguments(parser):
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--lsf-job-no', type=str, default='100', metavar='S',
                        help='Job-<ID> assigned by LSF.')
    parser.add_argument('--unique-port', type=str, default='100', metavar='S',
                        help='Which port to use, each client should have different value of this.')
    parser.add_argument('--heartbeats-timelimit', type=float, default=60, metavar='S',
                        help='time to issue heartbeats')
    parser.add_argument('--working-directory', type=str,
                        default='/cluster/scratch/biyuan/fetch_cache', metavar='S',
                        help='The IP of coordinator-server.')


def add_global_coordinator_arguments(parser):
    parser.add_argument('--db-server-address', type=str,
                        default="http://xzyao:agway-fondly-ell-hammer-flattered-coconut@db.yao.sh:5984/", metavar='N',
                        help='Key value store address.')


def add_torch_distributed_hybrid_inference_w_euler_coordinator_arguments(parser):
    parser.add_argument('--dist-backend', type=str, default='cupy_nccl', metavar='S',
                        help='backend type for distributed PyTorch (default: cupy_nccl)')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--lsf-job-no', type=str, default='100', metavar='S',
                        help='Job-<ID> assigned by LSF.')
    parser.add_argument('--unique-port', type=str, default='100', metavar='S',
                        help='Which port to use, each client should have different value of this.')
    parser.add_argument('--world-size', type=int, default=4, metavar='D',
                        help='world-size (default: 4)')
    parser.add_argument('--pipeline-group-size', type=int, default=4, metavar='D',
                        help='world-size (default: 4)')


def add_inference_details_arguments(parser):
    parser.add_argument('--model-name', type=str, default='./pretrained_models/gpt2', metavar='S',
                        help='trained model path')
    parser.add_argument('--model-type', type=str, default='gpt2', metavar='S',
                        help='trained model path')
    parser.add_argument('--infer-data', type=str, default='', metavar='S',
                        help='data path')
    parser.add_argument('--output-path', type=str, default=None, metavar='S',
                        help='output data path')
    parser.add_argument('--top-k', type=int, default=None, metavar='S',
                        help='sample from top k')
    parser.add_argument('--top-p', type=float, default=None, metavar='S',
                        help='sample from top p')
    parser.add_argument('--temperature', type=float, default=0, metavar='S',
                        help='temperature on logits')
    parser.add_argument('--token-micro-batch-size', type=int, default=2, metavar='S',
                        help='token generation micro batch size.')
    parser.add_argument('--prompt-micro-batch-size', type=int, default=1, metavar='S',
                        help='token generation micro batch size.')
    parser.add_argument('--echo-prompt', type=lambda x: (str(x).lower() == 'true'),
                        default=False, metavar='S',
                        help='append prompt to the generated text')
    parser.add_argument('--num-completions', type=int, default=1, metavar='S',
                        help='num of completions')
    parser.add_argument('--best-of', type=int, default=1, metavar='S',
                        help='num of best of completions')
    parser.add_argument('--top-k-per-token', type=int, default=0, metavar='S',
                        help='return top k candidate for each token')
    parser.add_argument('--budget', type=int, default=None, metavar='S',
                        help='budget: for each batch, auto-assign max(n_seq * n_tokens)')
    parser.add_argument('--stop', nargs='+', type=str, default=None,
                        help='stop words')
    parser.add_argument('--share-prefix', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='whether to share prefix of prompt')


def get_inference_arguments_str(args, add_rank=True, rank=None):
    arg_str = ''
    if args.fp16:
        arg_str += '_fp16'
    arg_str += '_b' + str(args.batch_size) + '_' + str(args.micro_batch_size)
    if hasattr(args, 'token_micro_batch_size'):
        arg_str += '_' + str(args.token_micro_batch_size)

    arg_str += '_s' + str(args.input_seq_length) + '_' + str(args.generate_seq_length)
    arg_str += '_p' + str(args.pipeline_group_size)
    if add_rank:
        if rank is not None:
            arg_str += '_' + str(rank)
        else:
            arg_str += '_' + str(args.rank)
    return arg_str


def get_hybrid_inference_arguments_str(args, add_rank=True, rank=None):
    arg_str = '_'
    arg_str += args.pp_mode
    if args.fp16:
        arg_str += '_fp16'
    arg_str += '_b' + str(args.prompt_micro_batch_size) + '_' + str(args.token_micro_batch_size)
    arg_str += '_s' + str(args.input_seq_length) + '_' + str(args.generate_seq_length)
    arg_str += '_gpu' + str(args.pipeline_group_size) + '_cpu' + str(args.world_size-args.pipeline_group_size)
    arg_str += '_pb' + str(args.producer_buffer_size) + '_cb' + str(args.consumer_buffer_size)
    if add_rank:
        if rank is not None:
            arg_str += '_' + str(rank)
        else:
            arg_str += '_' + str(args.rank)
    return arg_str


def print_arguments(args):
    args_dict = vars(args)
    print("======================Input Arguments=========================")
    for key in args_dict.keys():
        print(key, ": ", args_dict[key])
    print("==============================================================")
