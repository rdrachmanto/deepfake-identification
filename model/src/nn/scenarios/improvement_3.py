import os

import timm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import src.nn.config as config
from src.nn.training import iters


class XceptionNetImprovement3:
    def __init__(self) -> None:
        pass

    def _create_model_struct(self):
        model = timm.create_model(
            "hf_hub:timm/xception41.tf_in1k",
            pretrained=True,
            global_pool="max",
        )

        for param in model.parameters():
            param.requires_grad = True

        model.head.fc = nn.Sequential(
            nn.Linear(model.head.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        return model, loss_fn, optimizer

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        save_to: str,
        silent: bool,
    ):
        model, loss_fn, optimizer = self._create_model_struct()
        model.to(config.DEVICE)

        for t in range(epochs):
            iters.train(
                train_loader,
                model,
                loss_fn,
                optimizer,
                t,
                epochs,
                silent,
            )

        iters.test(test_loader, model, loss_fn, silent)

        if not os.path.exists(save_to):
            os.makedirs(save_to)

        torch.save(model, f"{save_to}/model.pth")
