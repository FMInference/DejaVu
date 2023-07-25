import torch
import numpy as np
import argparse
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from trainer_att import train

DATA = {
    "175b": {
        "c4": "../data/175b_c4",
    },
    "66b": {
        "c4": "../data/66b_c4",
    },
    "30b": {
        "c4": "../data/30b_c4",
    }
  
}

MODEL_CHOICES = ['175b', '66b', '30b']
DATA_CHOICES = ['c4']
CONFIG = {
    '175b':{
        'num_layer': 95,
        'ckt_storage': "bylayer",
        'd':12288,
        'h': 96,
        'N':400000,
    },
    '66b':{
        'num_layer': 64,
        'ckt_storage': "bylayer",
        'd':9216,
        'h': 72,
        'N':400000,
    },
    '30b':{
        'num_layer': 24,
        'ckt_storage': "bylayer",
        'd':2048,
        'h': 32,
        'N':400000,
    },
}

class BasicDataset(Dataset):
    def __init__(self, X, Y, n, train ):
        self.X = X
        self.Y = Y 
        self.n = n
        self.train = train

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.train:
            x = torch.Tensor(self.X[idx])
            y = torch.Tensor(self.Y[idx])
        else:
            x = torch.Tensor(self.X[-idx])
            y = torch.Tensor(self.Y[-idx])
        if y.sum()== 0:
            print("all zero y")
            exit()
        return x, y

def get_data(args, l):
    if CONFIG[args.model]['ckt_storage'] == "bylayer":
        path = f"{DATA[args.model][args.dataset]}/att_x_{l-1}.mmap"
        print(f"Reading query from {path}")
        query = np.array(np.memmap(path, dtype='float16', mode='r', shape=(400000,CONFIG[args.model]['d']))[: CONFIG[args.model]['N']])
    
        path = f"{DATA[args.model][args.dataset]}/score_norm_{l}.mmap"
        print(f"Reading attention label from {path}")
        label = np.array(np.memmap(path, dtype='float16', mode='r', shape=(400000,CONFIG[args.model]['h']))[: CONFIG[args.model]['N']])
        
        num_valid = (label.sum(-1) > 0).sum()
        print(num_valid)
        return  query[:num_valid], label[:num_valid]

def create_dataset(query, labels, args):
    total = len(query)
    num_train = int(0.95 * total)
    num_test = int(0.05 * total)

    print(f"Query shape: {query.shape}, Label shape: {labels.shape}")
    print(f"# training data: {num_train}, # test data: {num_test}")

    train_ds = BasicDataset(query, labels, num_train, True)
    test_ds = BasicDataset(query, labels, num_test, False)

    train_dataloader = DataLoader(
        train_ds, args.batch_size, shuffle=True, num_workers=0
    )
    test_dataloader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0)
    return train_dataloader, test_dataloader


def main():
    parser = argparse.ArgumentParser(description="PyTorch OPT Full Model")
    parser.add_argument("--model", type=str, default="66b", choices = MODEL_CHOICES)
    parser.add_argument("--dataset", type=str, default="c4", choices = DATA_CHOICES)
    parser.add_argument(
        "--L",
        type=int,
        default=0,
        help="which layer",
    )
    parser.add_argument(
        "--D",
        type=int,
        default=1000,
        help="low rank dimension",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=0.7,
        help="top k percent to mark as activate head",
    )
    args = parser.parse_args()

    print(args)
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 40, "Layer", args.L, "=" * 40)

    query, labels = get_data(args, args.L)
    train_loader, test_loader = create_dataset(query, labels, args)

    query_layer = torch.nn.Sequential(
        torch.nn.Linear(CONFIG[args.model]['d'], args.D, bias=None),
        torch.nn.Linear(args.D, CONFIG[args.model]['h'], bias=None),
    )
    
    print("Start Training")
    best_model, eval_result = train(
        query_layer,  train_loader, test_loader, args, device, verbal=True
    )

    path = f"../checkpoint/opt-{args.model}-sparse-predictor/{args.dataset}_att_k_{args.k}_layer{args.L}_-{eval_result['Recall']:.4f}-{eval_result['Classifier Sparsity']:.0f}.pt"
    torch.save(best_model, path)









if __name__ == "__main__":
    main()
