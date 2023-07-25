import torch
import argparse
from task_datasets.qqp import get_glue_qqp_train_data_loader
from task_datasets.tokenizer import build_tokenizer
from modules.gpt_modules import GlueSeqClassificationModel, GlueSeq2SeqClassificationModel
from deepspeed.profiling.flops_profiler import FlopsProfiler
from optimizer.optimizer import get_fp16_optimizer
from utils.dist_args_utils import add_qqp_task_arguments


def main():
    parser = argparse.ArgumentParser(description='Test Glue-qqp dataset')
    add_qqp_task_arguments(parser)
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--seq-length', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--embedding-dim', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--num-layers', type=int, default=3, metavar='N',
                        help='-')
    parser.add_argument('--num-heads', type=int, default=16, metavar='N',
                        help='-')
    parser.add_argument('--lr', type=float, default=0.01, metavar='N',
                        help='-')
    parser.add_argument('--loss-scale', type=float, default=64,
                        help='Static loss scaling, positive power of 2 values can improve fp16 convergence. ')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode.')
    parser.add_argument('--use-offload', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, we will offload the fp32 model to CPU RAM.')
    parser.add_argument('--task', type=str, default='Seq2SeqClassification', metavar='S',
                        help='What task to run?')
    args = parser.parse_args()
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')
    if args.fp16:
        print("<=== Training in fp16. ===>")
    tokenizer = build_tokenizer(args)
    print("token vocab size:", tokenizer.vocab_size)
    data_loader = get_glue_qqp_train_data_loader(args, tokenizer)
    num_classes = 2
    if args.task == 'SeqClassification':
        model = GlueSeqClassificationModel(args, tokenizer.vocab_size, num_classes, use_checkpoint=True).to(device)
    elif args.task == 'Seq2SeqClassification':
        model = GlueSeq2SeqClassificationModel(args, tokenizer.vocab_size, use_checkpoint=True).to(device)
    else:
        assert False
    print("Model info:")
    for name, param in model.named_parameters():
        print(name, ":", param.size())

    if args.fp16:
        model.half()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.fp16:
        optimizer = get_fp16_optimizer(args, optimizer, device)

    prof = FlopsProfiler(model)

    # for i in range(len(data_loader)):
    #    data = data_loader[i]
    # train_data_loader_iter = iter(data_loader)
    # print(next(train_data_loader_iter))
    for i, data in enumerate(data_loader):
        if i == 1:
            prof.start_profile()
        print("Check data:", data)
        input_ids = data['text'].to(device)
        # position_ids = get_position_id(args.seq_length, args.batch_size, device)
        if args.task == 'SeqClassification':
            labels = data['label'].to(device)

        elif args.task == 'Seq2SeqClassification':
            labels = data['text'].to(device)
            # shift_labels = labels[..., 1:].contiguous()

        optimizer.zero_grad(set_to_none=False)
        # output = model(input_ids, position_ids)

        # loss = loss_func(output, labels)
        if args.task == 'SeqClassification':
            output = model(input_ids)
            print(output.shape)
            loss = torch.nn.functional.cross_entropy(output, labels)
        elif args.task == 'Seq2SeqClassification':
            loss = model(input_ids, labels)
            # shift_logits = output[..., :-1, :].contiguous()
            # loss = torch.nn.functional.nll_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if i == 1:
            prof.stop_profile()
            flops = prof.get_total_flops()
            macs = prof.get_total_macs()
            params = prof.get_total_params()
            prof.print_model_profile()
            prof.end_profile()
            print("Flop raw: {}, PFlop: {} for a batch of 1024".format(flops, flops * 1024 / 10**15))
            print("Macs:", macs)
            print("Params:", params)

        loss.backward()
        optimizer.step()

        print("Iter ", i, "===== Loss: ", loss.item(), "======")
        if i > 3:
            break
        # print(data)


if __name__ == '__main__':
    main()
