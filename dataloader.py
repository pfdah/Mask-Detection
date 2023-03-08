import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloader(path, batch_size):
    '''Create the dataloader using the images in a folder

    Parameters
    ----------
    path : string
        the path of the folder
    batch_size : int
        the batch size of dataloader

    Returns
    -------
    dataloader object
        the created dataloader
    '''
    data_transforms = transforms.Compose([transforms.Resize((150,150)),
                                       transforms.ToTensor(),                                
                                       torchvision.transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225],
    ),
                                       ])
    data = datasets.ImageFolder(path,transform=data_transforms)

    
    return DataLoader(data, shuffle = True, batch_size=batch_size)
