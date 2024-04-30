import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import os
import numpy as np
import random

import torchvision
from torchvision.datasets import CIFAR100
from torchvision import transforms
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from par_resnet import ParResNet
from grouped_resnet import GroupedResNet


# Disgusting globals
model_dict = {
    'ParResNet': ParResNet,
    'GroupedResNet': GroupedResNet
}


# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "./data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup():
    # Function for setting the seed
    set_seed(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')

    # Create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)


def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)


class CIFARModule(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(
                self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer: \"{self.hparams.optimizer_name}\""

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[15], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log('test_acc', acc)


def train_model(model_name, save_name=None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run.
        save_name (optional) - checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name

    train_dataset = CIFAR100(root=DATASET_PATH, train=True, download=True)
    DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0,1,2))
    DATA_STD = (train_dataset.data / 255.0).std(axis=(0,1,2))
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEANS, DATA_STD)
    ])

    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32,32), scale=(0.8,1.0), ratio=(0.9,1.1)),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEANS, DATA_STD)
    ])

    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = CIFAR100(
        root=DATASET_PATH, train=True, transform=train_transform, download=True
    )
    val_dataset = CIFAR100(
        root=DATASET_PATH, train=True, transform=test_transform, download=True
    )
    set_seed(42)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    set_seed(42)
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

    # Loading the test set
    test_set = CIFAR100(
        root=DATASET_PATH, train=False, transform=test_transform, download=True
    )

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(
        train_set, batch_size=64, shuffle=True, drop_last=True, pin_memory=True, num_workers=4
    )
    val_loader = data.DataLoader(
        val_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4
    )
    test_loader = data.DataLoader(
        test_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4
    )

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
        accelerator="gpu",
        devices=1,
        max_epochs=20,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch")
        ],
        enable_progress_bar=True,
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = CIFARModule.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42) # To be reproducable
        model = CIFARModule(model_name=model_name, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = CIFARModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result


def main():
    setup()
    model, results = train_model(
        model_name="GroupedResNet",
        model_hparams={
            "in_channels": 3,
            "classes": 100,
            "s0_depth": 10,
            "s1_depth": 20,
            "s2_depth": 10,
            "groups": 1
        },
        optimizer_name="Adam",
        optimizer_hparams={
            "lr": 0.1,
        }
    )


if __name__ == '__main__':
    main()
