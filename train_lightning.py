from enum import Enum
from typing import Optional, Any

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
import os

from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from par_resnet import ParResNet, ParParResNet
from grouped_resnet import GroupedResNet
from split_resnet import SplitResNet

# Disgusting globals
model_dict = {
    'ParResNet': ParResNet,
    'ParParResNet': ParParResNet,
    'GroupedResNet': GroupedResNet,
    'SplitResNet': SplitResNet
}


# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_ROOT = "./data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/"


class TrainDuration(Enum):
    SHORT = "short"
    DEFAULT = "default"
    LONG = "long"


def setup():
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    #torch.set_default_device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')

    # Create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)


def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)


class DelayedStartEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set start_epoch to None or 0 for no delay
        self.start_epoch = start_epoch

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if (self.start_epoch is not None) and (trainer.current_epoch < self.start_epoch):
            return
        super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if (self.start_epoch is not None) and (trainer.current_epoch < self.start_epoch):
            return
        super().on_validation_end(trainer, pl_module)


from bisect import bisect_right


class SeqLR(optim.lr_scheduler.SequentialLR):
    def step(self, metric):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            scheduler.step(0)
        else:
            scheduler.step(metric)

        self._last_lr = scheduler.get_last_lr()


class ResNetModule(pl.LightningModule):
    def __init__(self, model_name, model_hparams, duration=TrainDuration.DEFAULT, **kwargs):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary.
        """
        super().__init__()
        self.duration = duration
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 299, 299))
        self.warmup_scheduler = None
        self.reset_lr = None

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams)

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
        # scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[15, 30], gamma=0.1)
        self.warmup_scheduler = optim.lr_scheduler.ConstantLR(optimizer, 0.1)
        self.reset_lr = optim.lr_scheduler.ConstantLR(optimizer, 10)

        if self.duration == TrainDuration.SHORT:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], 0.25)
        elif self.duration == TrainDuration.DEFAULT:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.2,
                patience=3,
                threshold_mode='abs'
            )

        elif self.duration == TrainDuration.LONG:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            raise Exception(f"Unknown duration value: {self.duration}")

        freq = 1
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "frequency": freq
            }
        }

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]) -> None:
        print(self.current_epoch)
        if self.current_epoch < 4:
            self.warmup_scheduler.step()
        elif self.current_epoch == 5:
            self.reset_lr.step()
        else:
            super().lr_scheduler_step(scheduler, metric)

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_loss', loss, sync_dist=True)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (labels == preds.argmax(dim=-1)).float().mean()

        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log('test_acc', acc, sync_dist=True)


def train_model(
    model_name,
    dataset="imagenet",
    workers=0,
    save_name=None,
    nodes=2,
    duration=TrainDuration.DEFAULT,
    **kwargs
):
    """
    Inputs:
        model_name - Name of the model you want to run.
        save_name (optional) - checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_set = ImageFolder(
        root=os.path.join(DATASET_ROOT, dataset, "train"), transform=train_transform
    )
    test_set = ImageFolder(
        root=os.path.join(DATASET_ROOT, dataset, "val"), transform=test_transform
    )
    train_set, val_set = torch.utils.data.random_split(train_set, [9/10, 1/10])

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(
        train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=workers
    )
    val_loader = data.DataLoader(
            val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=workers
    )
    test_loader = data.DataLoader(
        test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=workers
    )
    logger = TensorBoardLogger(
        save_dir=os.path.join(CHECKPOINT_PATH, save_name, dataset),
        name=f""
             f"{kwargs['model_hparams']['groups']}_"
             f"{kwargs['model_hparams']['s0_depth']}_"
             f"{kwargs['model_hparams']['s1_depth']}_"
             f"{kwargs['model_hparams']['s2_depth']}_{duration}",
        default_hp_metric=False,
        log_graph=True
    )
    stop = 40
    if duration == TrainDuration.SHORT:
        epochs = 15
    elif duration == TrainDuration.DEFAULT:
        epochs = 80
    elif duration == TrainDuration.LONG:
        epochs = 200
        stop = 120
    else:
        raise Exception(f"Unknown duration value: {duration}")

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name, dataset),
        accelerator="gpu",
        devices="auto",
        num_nodes=nodes,
        max_epochs=epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
            DelayedStartEarlyStopping(
                start_epoch=stop,
                monitor="val_acc",
                patience=5,
                min_delta=0.001,
                verbose=False,
                mode="max"
            )
        ],
        enable_progress_bar=True,
        precision="bf16-mixed",
        logger=logger
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard

    # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    # if os.path.isfile(pretrained_filename):
    #     print(f"Found pretrained model at {pretrained_filename}, loading...")
    #     model = ResNetModule.load_from_checkpoint(pretrained_filename)
    # else:
    kwargs['duration'] = duration
    kwargs['dataset'] = dataset
    model = ResNetModule(model_name=model_name, **kwargs)
    trainer.fit(model, train_loader, val_loader)
    model = ResNetModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    test_result = trainer.test(model, test_loader, verbose=False)

    return model, test_result


def main():
    parser = argparse.ArgumentParser(description="Train a model with specified hyperparameters")
    parser.add_argument(
        "--groups",
        type=int,
        default=1,
        help="Number of groups for the model"
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of workers for the dataloader"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="path to the dataset (relative to DATASET_ROOT)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="GroupedResNet",
        help="path to the dataset (relative to DATASET_ROOT)"
    )

    parser.add_argument(
        "--s0",
        type=int,
        default=10,
        help="path to the dataset (relative to DATASET_ROOT)"
    )

    parser.add_argument(
        "--s1",
        type=int,
        default=20,
        help="path to the dataset (relative to DATASET_ROOT)"
    )

    parser.add_argument(
        "--s2",
        type=int,
        default=10,
        help="path to the dataset (relative to DATASET_ROOT)"
    )

    parser.add_argument(
        "--duration",
        type=TrainDuration,
        choices=list(TrainDuration),
        default=TrainDuration.DEFAULT,
        help="Training duration: short, default, or long"
    )

    args = parser.parse_args()


    if args.duration == TrainDuration.SHORT:
        lr = 0.05
    elif args.duration == TrainDuration.DEFAULT:
        lr = 0.025
    elif args.duration == TrainDuration.LONG:
        lr = 0.1
    else:
        raise Exception(f"Unknown duration value: {args.duration}")

    setup()
    model, results = train_model(
        model_name=args.model,
        nodes=args.nodes,
        workers=args.workers,
        dataset=args.dataset,
        model_hparams={
            "in_channels": 3,
            "classes": 100,
            "s0_depth": args.s0,
            "s1_depth": args.s1,
            "s2_depth": args.s2,
            "groups": args.groups,
        },
        optimizer_name="Adam",
        optimizer_hparams={
            "lr": lr,
        },
        duration=args.duration
    )


if __name__ == '__main__':
    main()
