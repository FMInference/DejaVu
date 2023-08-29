import torch
import copy
import numpy as np
from tqdm import tqdm


def eval_print(validation_results):
    result = ""
    for metric_name, metirc_val in validation_results.items():
        result = f"{result}{metric_name}: {metirc_val:.4f} "
    return result


def generate_label(y):
    # positive
    one_hot = (y > 0).to(y.dtype)
    return one_hot


def evaluate(model, device, loader, args, smalltest=False):
    model.eval()

    eval = {
        "Loss": [],
        "Loss Weight": [],
        "Recall": [],
        "Classifier Sparsity": [],
        "True Sparsity": [],
    }
    with torch.no_grad():
        for batch_idx, batch in enumerate((loader)):
            x, y = batch
            y = y.float().to(device)
            y = generate_label(y)
            logits = model(x.to(device))
            probs = logits.sigmoid()
            preds = probs >= 0.5

            dif = y.int() - preds.int()
            miss = dif > 0.0  # classifier didn't activated target neuron

            weight = (y.sum() / y.numel()) + 0.005
            loss_weight = y * (1 - weight) + weight
            eval["Loss Weight"] += [weight.item()]
            eval["Loss"] += [
                torch.nn.functional.binary_cross_entropy(
                    probs, y, weight=loss_weight
                ).item()
            ]

            eval["Recall"] += [
                ((y.sum(dim=1).float() - miss.sum(dim=1).float()).mean().item())
            ]
            eval["True Sparsity"] += [y.sum(dim=1).float().mean().item()]
            eval["Classifier Sparsity"] += [preds.sum(dim=1).float().mean().item()]

            if batch_idx >= 100 and smalltest:
                break

    for k, v in eval.items():
        eval[k] = np.array(v).mean()

    eval["Recall"] = eval["Recall"] / eval["True Sparsity"]
    return eval


def train(model, train_loader, valid_loader, args, device, verbal=True):
    num_val = 0
    early_stop_waiting = 5
    val_inter = len(train_loader) // (num_val + 1) + 1
    num_print = 0
    print_inter = len(train_loader) // (num_print + 1) + 1
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01)

    eval_results = evaluate(model, device, valid_loader, args, smalltest=True)
    if verbal:
        print(f"[Start] {eval_print(eval_results)}")

    best_model = copy.deepcopy(model.state_dict())
    base_acc = eval_results["Recall"]
    best_eval = eval_results
    no_improve = 0

    for e in range(args.epochs):
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            model.train()
            x, y = batch
            optimizer.zero_grad()

            y = y.float().to(device)
            y = generate_label(y)
            logits = model(x.to(device))
            probs = logits.sigmoid()

            weight = (y.sum() / y.numel()) + 0.005
            loss_weight = y * (1 - weight) + weight
            loss = torch.nn.functional.binary_cross_entropy(
                probs, y, weight=loss_weight
            )
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % print_inter == 0:
                print(
                    f"[ {e}, {batch_idx}] Loss: {loss.item():.4f}, Loss weight: { weight.item():.4f}"
                )

            if (
                (batch_idx + 1) % val_inter == 0
                and batch_idx != 0
                and batch_idx != len(train_loader)
            ):
                valid_eval_results = evaluate(
                    model, device, valid_loader, smalltest=False
                )
                train_eval_results = evaluate(
                    model, device, train_loader, smalltest=True
                )
                model.train()
                print(
                    f"[{e}, {batch_idx}] Validation: {eval_print(valid_eval_results)}"
                )
                print(f"[{e}, {batch_idx}] Train: {eval_print(train_eval_results)}")

        train_eval_results = evaluate(model, device, train_loader, args, smalltest=True)
        epoch_eval_results = evaluate(
            model, device, valid_loader, args, smalltest=False
        )
        if verbal:
            print(f"[Epoch {e+1}] [Train] {eval_print(train_eval_results)}")
            print(f"[Epoch {e+1}] [Valid] {eval_print(epoch_eval_results)}\n")

        if epoch_eval_results["Recall"] > base_acc:
            base_acc = epoch_eval_results["Recall"]
            best_eval = epoch_eval_results
            model.cpu()
            best_model = copy.deepcopy(model.state_dict())
            model.to(device)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop_waiting or base_acc > 0.99:
            break

    return best_model, best_eval
