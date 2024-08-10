import os

import timm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import model.src.nn.config as config
from model.src.nn.training import iters


class XceptionNetImprovement2:
    def __init__(self) -> None:
        pass

    def _create_pretrained_model(self):
        model = timm.create_model("hf_hub:timm/xception41.tf_in1k", pretrained=True)

        for param in model.parameters():
            param.requires_grad = True

        model.head.fc = nn.Linear(model.head.fc.in_features, 2)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        return model, loss_fn, optimizer

    def _create_model_to_finetune(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False

        model.head.fc = nn.Sequential(
            nn.Linear(model.head.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.Linear(256, 2),
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        return model, loss_fn, optimizer

    def pretrain(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        silent: bool,
    ):
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

        return model

    def fine_tune(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        save_to: str,
        silent: bool,
    ):
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

    def train(
        self,
        pretrain_train_loader: DataLoader,
        pretrain_test_loader: DataLoader,
        pretraining_epochs: int,
        fine_tuning_train_loader: DataLoader,
        fine_tuning_test_loader: DataLoader,
        fine_tuning_epochs: int,
        save_to: str,
        silent: bool,
    ):
        model, loss_fn, optimizer = self._create_pretrained_model()
        model.to(config.DEVICE)

        pretrained_model = self.pretrain(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=pretrain_train_loader,
            test_loader=pretrain_test_loader,
            epochs=pretraining_epochs,
            silent=silent,
        )

        pretrained_model, loss_fn, optimizer = self._create_model_to_finetune(
            pretrained_model
        )
        pretrained_model.to(config.DEVICE)

        self.fine_tune(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=fine_tuning_train_loader,
            test_loader=fine_tuning_test_loader,
            epochs=fine_tuning_epochs,
            save_to=save_to,
            silent=silent
        )
