import os

import timm
import torch
from torch import nn, optim

import src.nn.config as config
from src.nn.training import iters


class XceptionNetBaseline:
    def __init__(self) -> None:
        pass

    def _create_pretrained_model(self):
        model = timm.create_model("hf_hub:timm/xception41.tf_in1k", pretrained=True)

        for param in model.parameters():
            param.requires_grad = True

        model.head.fc = nn.Sequential(
            nn.Linear(model.head.fc.in_features, 2),
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        return model, loss_fn, optimizer

    def _create_model_to_finetune(self, model):
        for param in model.parameters():
            param.requires_grad = False

        model.head.fc = nn.Sequential(
            nn.Linear(model.head.fc.in_features, 2),
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        return model, loss_fn, optimizer

    def pretrain(
        self, model, loss_fn, optimizer, train_loader, test_loader, epochs: int
    ):
        for t in range(epochs):
            iters.train(train_loader, model, loss_fn, optimizer, t, epochs)

        iters.test(test_loader, model, loss_fn)

        return model

    def fine_tune(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        test_loader,
        epochs: int,
        save_to: str,
    ):
        for t in range(epochs):
            iters.train(train_loader, model, loss_fn, optimizer, t, epochs)

        iters.test(test_loader, model, loss_fn)

        if not os.path.exists(save_to):
            os.makedirs(save_to)

        torch.save(model.state_dict(), f"{save_to}/model.pth")

    def train(
        self,
        pretrain_train_loader,
        pretrain_test_loader,
        pretraining_epochs: int,
        fine_tuning_train_loader,
        fine_tuning_test_loader,
        fine_tuning_epochs: int,
        save_to: str,
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
        )
