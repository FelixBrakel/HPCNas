import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from par_resnet import ParResNet, MoEResNet
from naslib.utils import setup_logger
from naslib.utils.log import log_every_n_seconds


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.version.cuda)
    # Hyperparameters
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.025

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, transform=valid_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # Model
    model = MoEResNet(classes=100)  # Change num_classes to 10 for CIFAR-10
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    logger = setup_logger("log.log")
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    # Training loop
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Backward and optimize
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss = scaler.scale(loss)

            log_every_n_seconds(
                logging.INFO,
                f"Epoch {epoch}-{i}, Train loss: {loss:.5f}, Learning rate: {scheduler.get_last_lr():}",
                n=5,
                name='naslib'
            )

            loss.backward()
            # scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()

        scheduler.step()


# Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

    # Save the model checkpoint
    # torch.save(model.state_dict(), 'inception_resnet_v2_cifar10.ckpt')


if __name__ == '__main__':
    main()
