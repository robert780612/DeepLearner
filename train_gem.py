import os
import time
import random
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageFile

import pretrainedmodels
import torchvision.transforms.functional as TF
from models import get_se_resnet50_gem

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('train_image_path', help='training folder')
parser.add_argument('train_csv_path', help='training label')
parser.add_argument('device', help='cuda')
parser.add_argument('--img_size', defualt=256, type=int)
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--model_path')
args = parser.parse_args()
train_image_path = args.train_image_path
train_csv_path = args.train_csv_path
device = args.device


class RetinopathyDatasetTrain2015(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(train_image_path, self.data.loc[idx, 'image'] + '.jpeg')
        image = Image.open(img_name)
        image = image.resize((args.img_size, args.img_size), resample=Image.BILINEAR)

        if self.transforms is not None:
            image = self.transforms(image)
            norm = transforms.Compose([transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
        label = torch.tensor(self.data.loc[idx, 'level'])
        return {'image': norm(image),
                'labels': label}


class RetinopathyDatasetTrain(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(train_image_path, self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = image.resize((args.img_size, args.img_size), resample=Image.BILINEAR)

        if self.transforms is not None:
            image = self.transforms(image)
            norm = transforms.Compose([transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])
        return {'image': norm(image),
                'labels': label
               }

        
"""
contrast_range=0.2,
brightness_range=20.,
hue_range=10.,
saturation_range=20.,
blur_and_sharpen=True,
rotate_range=180.,
scale_range=0.2,
shear_range=0.2,
shift_range=0.2,
do_mirror=True,
"""

class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


# def gem(x, p=3, eps=1e-6):
#     return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


# class GeM(nn.Module):
#     def __init__(self, p=3, eps=1e-6):
#         super(GeM,self).__init__()
#         self.p = Parameter(torch.ones(1)*p)
#         self.eps = eps
#     def forward(self, x):
#         return gem(x, p=self.p, eps=self.eps)       
#     def __repr__(self):
#         return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'



if __name__=="__main__":
    model = get_se_resnet50_gem(pretrain='imagenet')
    model.load_state_dict(torch.load(pretrain, map_location='cuda:0'))
    model = model.to(device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
    print(model)

    ### Dataset
    train_transforms = torch.nn.Sequential(
        transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(3, sigma=(0.1, 2.0))]), p=0.5),
        transforms.RandomAffine(180, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
    )
    if args.pretrain == True:
        train_dataset = RetinopathyDatasetTrain2015(csv_file=train_csv_path, transforms=train_transforms)
    else:
        train_dataset = RetinopathyDatasetTrain(csv_file=train_csv_path, transforms=train_transforms)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    ### Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    # optimizer = optim.SGD(lr=1e-3, momentum=0.9, nesterov=True)

    ### Train
    since = time.time()
    criterion = nn.SmoothL1Loss()
    num_epochs = 30
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        model.train()
        running_loss = 0.0
        #tk0 = tqdm(data_loader, total=int(len(data_loader)))
        counter = 0
        for bi, d in enumerate(data_loader):
            inputs = d["image"]
            labels = d["labels"].view(-1, 1)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            counter += 1
            #tk0.set_postfix(loss=(running_loss / (counter * data_loader.batch_size)))
            #epoch_loss = running_loss / len(data_loader.dataset)
            if bi % 20 == 0:
                print('Training Loss: {}, {}, {:.4f}'.format(epoch, bi, running_loss / counter), flush=True)

        # save model every 15 epochs
        if args.pretrain:
            torch.save(model.state_dict(), f"pretrain_model{epoch}.pth")
        else:
            if epoch % 15 == 0:
                torch.save(model.state_dict(), f"model{epoch}.pth")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))




