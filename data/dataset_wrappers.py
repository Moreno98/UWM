import torchvision.datasets as torchvision_datasets
import os
import torch
import utils.datasets as dt
import kaggle

def get_caltech_dataset(
    root,
    split,
    transform,
    download,
):
    return torchvision_datasets.Caltech101(
        root = root,
        target_type = 'category',
        transform = transform,
        download = download
    )

def get_dtd_dataset(
    root,
    split,
    transform,
    download,
):
    datasets = []
    for partition in range(1, 11):
        datasets.append(
            torchvision_datasets.DTD(
                root = root,
                split = split,
                partition = partition,
                transform = transform,
                download = download
            )
        )
    return torch.utils.data.ConcatDataset(datasets)

def get_aircraft_dataset(
    root,
    split,
    transform,
    download,
):
    return torchvision_datasets.FGVCAircraft(
        root = root,
        split = split,
        transform = transform,
        download = download
    )

def get_sun_dataset(
    root,
    split,
    transform,
    download,
):
    return torchvision_datasets.SUN397(
        root = root,
        transform = transform,
        download = download
    )

def get_food_dataset(
    root,
    split,
    transform,
    download,
):
    return torchvision_datasets.Food101(
        root = root,
        split = split,
        transform = transform,
        download = download
    )

def get_oxford_pets_dataset(
    root,
    split,
    transform,
    download,
):
    return torchvision_datasets.OxfordIIITPet(
        root = root,
        split = split,
        transform = transform,
        download = download
    )

def get_ucf101_dataset(
    root,
    split,
    transform,
    download,
):
    return dt.UCF_101(
        root = root,
        train = False,
        transform = transform
    )

def get_flowers102_dataset(
    root,
    split,
    transform,
    download,
):
    return torchvision_datasets.Flowers102(
        root = root,
        split = split,
        transform = transform,
        download = download
    )

def get_cifar10_dataset(
    root,
    split,
    transform,
    download,
):
    return torchvision_datasets.CIFAR10(
        root = root,
        train = False,
        transform = transform,
        download = download
    )

def get_cifar100_dataset(
    root,
    split,
    transform,
    download,
):
    return torchvision_datasets.CIFAR100(
        root = root,
        train = False,
        transform = transform,
        download = download
    )

def get_mnist_dataset(
    root,
    split,
    transform,
    download,
):
    return torchvision_datasets.MNIST(
        root = root,
        train = False,
        transform = transform,
        download = download
    )

def get_eurosat_dataset(
    root,
    split,
    transform,
    download,
):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    return torchvision_datasets.EuroSAT(
        root = root,
        transform = transform,
        download = download
    )

def get_imagenet_dataset(
    root,
    split,
    transform,
    download,
):
    return torchvision_datasets.ImageNet(
        root = root,
        split = 'val',
        transform = transform
    )


def get_imagenetV2_dataset(
    root,
    split,
    transform,
    download,
):
    return torchvision_datasets.ImageFolder(
        root=root,
        transform=transform
    )

def get_imagenetR_dataset(
    root,
    split,
    transform,
    download,
):
    return torchvision_datasets.ImageFolder(
        root=root,
        transform=transform
    )

def get_imagenetA_dataset(
    root,
    split,
    transform,
    download,
):
    return torchvision_datasets.ImageFolder(
        root=root,
        transform=transform
    )

def get_imagenetSketch_dataset(
    root,
    split,
    transform,
    download,
):
    return torchvision_datasets.ImageFolder(
        root=root,
        transform=transform
    )

def get_standfordcars_dataset(
    root,
    split,
    transform,
    download,
):  
    # if root empty
    if os.path.exists(root) and len(os.listdir(root)) == 0 or not os.path.exists(root):
        kaggle.api.dataset_download_files('rickyyyyyyy/torchvision-stanford-cars', path=root, unzip=True)
    return torchvision_datasets.StanfordCars(
        root = root,
        split = split,
        transform = transform,
        download = False
    )