import torch
import argparse
from task_datasets.qqp import QQPDataset
from task_datasets.tokenizer import build_tokenizer


def train_data_loader(args, tokenizer):
    train_dataset = QQPDataset('training', args.train_data, tokenizer, args.seq_length)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.micro_batch_size,
                                                    sampler=train_sampler,
                                                    shuffle=False,
                                                    num_workers=4,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader



def main():
    parser = argparse.ArgumentParser(description='Test Glue-qqp dataset')
    parser.add_argument('--train-data', nargs='+', default=['./task_datasets/data/QQP/train.tsv'], metavar='S',
                        help='path to the training data')
    parser.add_argument('--valid-data', nargs='+', default=['./task_datasets/data/QQP/test.tsv'], metavar='S',
                        help='path to the training data')
    parser.add_argument('--seq-length', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--micro-batch-size', type=int, default=2, metavar='N',
                        help='-')
    parser.add_argument('--tokenizer-type', type=str, default='BertWordPieceLowerCase', metavar='S',
                        help='which tokenizer to use.')
    parser.add_argument('--vocab-file', type=str, default='./task_datasets/data/bert-large-cased-vocab.txt', metavar='S',
                        help='which tokenizer to use.')
    parser.add_argument('--vocab-extra-ids', type=int, default=0, metavar='N',
                        help='-')
    parser.add_argument('--make-vocab-size-divisible-by', type=int, default=128, metavar='N',
                        help='-')
    args = parser.parse_args()
    tokenizer = build_tokenizer(args)
    data_loader = train_data_loader(args, tokenizer)
    data_iter = iter(data_loader)
    for i in range(len(data_iter)):
        batch = next(data_iter)
        print(batch)
    # for data in data_loader:
    #    print(data)


if __name__ == '__main__':
    main()