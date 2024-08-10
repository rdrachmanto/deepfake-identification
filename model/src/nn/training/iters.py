import torch
from torch.utils.data import DataLoader
from torch import nn

from tqdm import tqdm

import model.src.nn.config as config


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    num_epochs: int,
    silent: bool = False,
):
    model.train()

    # for progress bar
    bar = tqdm(
        dataloader,
        desc=f"{epoch+1}/{num_epochs}",
        dynamic_ncols=True,
        bar_format=config.BAR_FORMAT,
        disable=silent,
    )
    running_loss = 0.0

    for batch, (x, y) in enumerate(bar):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Compute prediction error
        pred = model(x)  # This is calling the forward() function in the model
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()  # gradients are computed here
        optimizer.step()  # weights and biases are updated, using the gradients from loss.backward()
        optimizer.zero_grad()  # reset gradients to zero

        # Update running loss
        running_loss += loss.item()
        current_loss = running_loss / (batch + 1)

        # Update progress bar with info
        bar.set_postfix(loss="{:.3f}".format(current_loss))


def test(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, silent: bool = False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    bar = tqdm(
        dataloader,
        desc="Eval",
        dynamic_ncols=True,
        bar_format=config.BAR_FORMAT,
        disable=silent,
    )

    with torch.no_grad():
        for batch, (x, y) in enumerate(bar):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
