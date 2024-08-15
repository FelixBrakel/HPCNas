from enum import Enum

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
from par_resnet import ParResNet
from par_par_resnet import ParParResNet
from grouped_resnet import GroupedResNet
from split_resnet import SplitResNet
from base_resnet import BaseResNet
from resnet import ResNet
from base_resnet_768 import BaseResNet as BaseResNet768
from base_resnet_768_half import BaseResNet as BaseResNetHalf
from base_resnet_true_half import BaseResNet as BaseResNetTrueHalf
from base_resnet_half_quart import BaseResNet as BaseResNetHalfQuart
from base_resnet_quart_quart import BaseResNetQuartQuart
from base_resnet_1_8 import BaseResNetEighth

model_dict = {
    'ParResNet': ParResNet,
    'ParParResNet': ParParResNet,
    'GroupedResNet': GroupedResNet,
    'SplitResNet': SplitResNet,
    'BaseResNet': BaseResNet,
    'BaseResNet768': BaseResNet768,
    'BaseResNetHalf': BaseResNetHalf,
    'BaseResNetTrueHalf': BaseResNetTrueHalf,
    'BaseResNetHalfQuart': BaseResNetHalfQuart,
    'BaseResNetQuartQuart': BaseResNetQuartQuart,
    'BaseResNetEighth': BaseResNetEighth,
    'ResNet': ResNet,
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        torch.compiler.cudagraph_mark_step_begin()
        return self.model(imgs)

    def configure_optimizers(self):
        # legacy, Adam is not used anymore
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(
                self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer: \"{self.hparams.optimizer_name}\""

        if self.duration == TrainDuration.LONG:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                threshold_mode='abs',
                threshold=0.002,
                factor=0.84,
                patience=1,
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, 2, 0.94)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc"
            }
        }

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
        self.log('val_loss', loss, sync_dist=True)
        acc = (labels == preds.argmax(dim=-1)).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc, sync_dist=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('test_acc', acc, sync_dist=True)



def train_model(
    model_name,
    dataset="imagenet",
    seed=42,
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

    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Loading the training dataset. We need to split it into a training and validation part
    train_set = ImageFolder(
        root=os.path.join(DATASET_ROOT, dataset, "train"), transform=train_transform
    )

    generator = torch.Generator().manual_seed(seed)
    # generator = None
    train_set, test_set, val_set = torch.utils.data.random_split(
        train_set, [70/100, 20/100, 10/100], generator=generator
    )

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(
        train_set, batch_size=256, shuffle=True, drop_last=True, num_workers=workers,
        generator=generator, pin_memory=True
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
    )

    if duration == TrainDuration.SHORT:
        stop = 10
        epochs = 40
    elif duration == TrainDuration.DEFAULT:
        epochs = 60
        stop = 60
    elif duration == TrainDuration.LONG:
        epochs = 70
        stop = 40
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
                patience=9,
                min_delta=0.01,
                verbose=False,
                mode="max"
            )
        ],
        enable_progress_bar=True,
        precision="bf16-mixed",
        logger=logger
    )

    kwargs['duration'] = duration
    kwargs['dataset'] = dataset
    model = ResNetModule(model_name=model_name, **kwargs).cuda()
    model = torch.compile(model, options={"shape_padding": True})

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
        "--seed",
        type=int,
        default=42,
        help="Seed for randomness"
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
        lr = 0.0025
    elif args.duration == TrainDuration.DEFAULT:
        lr = 0.045
    elif args.duration == TrainDuration.LONG:
        lr = 0.045
    else:
        raise Exception(f"Unknown duration value: {args.duration}")

    setup()
    model, results = train_model(
        model_name=args.model,
        nodes=args.nodes,
        workers=args.workers,
        dataset=args.dataset,
        seed=args.seed,
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
