from torchvision import transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_train_transform():
    """
    Transform used for search/training data.

    Random augmentation:
      1. RandomCrop(32, padding=4)
      2. RandomHorizontalFlip()

    Preprocessing:
      3. ToTensor()
      4. Normalize()
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD,
        ),
    ])


def get_cifar10_val_transform():
    """
    Transform used for validation data.

    Important:
    Validation must NOT contain random augmentation.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD,
        ),
    ])