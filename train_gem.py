import os
import time
import random
import json

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageFile

import pretrainedmodels
import torchvision.transforms.functional as TF
from models import get_se_resnet50_gem

TRAIN_IMAGE_PATH = './train_images'
WORK_DIR = 'work_dir/448'
LOAD_MODEL = os.path.join(WORK_DIR,'latest.pth')

device = torch.device("cuda")
ImageFile.LOAD_TRUNCATED_IMAGES = True

if not os.path.isdir(WORK_DIR):
    os.mkdir(WORK_DIR)

class RetinopathyDatasetTrain(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(TRAIN_IMAGE_PATH, self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = image.resize((448, 448), resample=Image.BILINEAR)

        if self.transforms is not None:
            image = self.transforms(image)

        label = torch.tensor(self.data.loc[idx, 'diagnosis'])
        return {'image': transforms.ToTensor()(image),
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

def split_data(dataset, file_name = 'split.json'):
    path = os.path.join(WORK_DIR, file_name)
    if os.path.isfile(path):
        with open(path, 'r') as f:
            return json.load(f)

    total_class_num = {0: 1805, 1: 370, 2: 999, 3: 193, 4: 295}
    train_class_num = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    val_class_num = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    print('Dataset distribution: {}'.format(total_class_num))
    print('Splitting...')
    train_set = []
    val_set = []

    for i in range(len(train_dataset)):
        data = int(train_dataset[i]['labels'])
        if random.randint(0, 10)>0:
            train_class_num[data] += 1
            train_set.append(i)
        else:
            val_class_num[data] += 1
            val_set.append(i)

    print('Training set distribution: {}'.format(train_class_num))
    print('Validation set distribution: {}'.format(val_class_num))

    split_dict = {'train':train_set, 'val':val_set}
    with open(path, 'w') as f:
        json.dump(split_dict, f)
    return split_dict

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__=="__main__":

    ### Model
    # model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet')
    # model.avg_pool = GeM()
    # model.last_linear = nn.Linear(2048, 1)

    model = get_se_resnet50_gem(pretrain=None)#'imagenet')
    model = model.to(device)
    print(model)
    if os.path.isfile(LOAD_MODEL):
        model.load_state_dict(torch.load(LOAD_MODEL))
    # model.eval()
    ### Dataset
    train_transforms = torch.nn.Sequential(
        transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(3, sigma=(0.1, 2.0))]), p=0.5),
        transforms.RandomAffine(180, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
        transforms.RandomHorizontalFlip(p=0.5)
    )
    train_dataset = RetinopathyDatasetTrain(csv_file='./train.csv', transforms=train_transforms)
    
    split_dict = split_data(train_dataset)
    tr_sampler = SubsetRandomSampler(split_dict['train'])
    val_sampler = SubsetRandomSampler(split_dict['val'])

    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=tr_sampler, num_workers=0)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=val_sampler, num_workers=0)

    ### Optimizer
    # optimizer = optim.Adam(model.parameters(), lr=0.0002)
    optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-5)
    stepLR = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.2)

    ### Train
    since = time.time()
    criterion = nn.SmoothL1Loss()
    num_epochs = 120
    for epoch in range(num_epochs):
        model.train()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        model.train()
        running_loss = 0.0
        tk0 = tqdm(data_loader, total=int(len(data_loader)),ncols=150)
        counter = 0
        correct = 0
        for bi, d in enumerate(tk0):
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
            correct += (torch.round(outputs) == labels).sum().item()
            running_loss += loss.item() * inputs.size(0)
            counter += inputs.shape[0]
            tk0.set_postfix(loss=running_loss/counter)
        stepLR.step()
        epoch_loss = running_loss / counter
        print('Training Loss: {:.4f}, Accuracy:{:.4}%, lr:{:.4}'.format(epoch_loss, 100.*correct/counter, get_lr(optimizer)))

        # save model every 15 epochs
        if epoch % 4 == 0:
            train_dataset.transforms = torch.nn.Sequential()
            torch.save(model.state_dict(), os.path.join(WORK_DIR,f"model_epoch{epoch+1}.pth"))
            torch.save(model.state_dict(), os.path.join(WORK_DIR,"latest.pth"))
            model.eval()
            running_loss = 0.0
            correct = 0.0
            tk0 = tqdm(val_loader, total=int(len(val_loader)),ncols=100)
            counter = 0
            for bi, d in enumerate(tk0):
                inputs = d["image"]
                labels = d["labels"].view(-1, 1)
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                correct += (torch.round(outputs) == labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                counter += inputs.shape[0]
                tk0.set_postfix(loss=running_loss/counter)
            epoch_loss = running_loss / counter
            show_str = 'Evaluation Loss: {:.4f}, Accuracy: {:.4f}%'.format(epoch_loss, 100.*correct/counter)
            print(show_str)
            with open(os.path.join(WORK_DIR,"log.txt"),'a') as f:
                f.write("epoch {}   {}\n".format(epoch+1,show_str))
            train_dataset.transforms = train_transforms

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))




