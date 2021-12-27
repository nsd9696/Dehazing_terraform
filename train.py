from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from ipt import ImageProcessingTransformer
from functools import partial
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='Dense_Haze_NTIRE19')
parser.add_argument('--pretrained_model', type=str, default='model_1016_0.2')
parser.add_argument('--epoch', type=int, default=300)

args = parser.parse_args()
data_path = args.data_path
pretrained_model = args.pretrained_model
nb_epochs = args.epoch


class ImageProcessDataset(Dataset):
    def __init__(self, data_dir, transform):

        # split data_dir to train, train_label
        train_path = data_dir + '/' + 'hazy'
        label_path = data_dir + '/' + 'target'

        try:
            train_list = sorted(os.listdir(train_path))
            label_list = sorted(os.listdir(label_path))
        except:
            raise ValueError

        train_list = [data_dir + '/' + 'hazy' + '/' + i for i in train_list]
        label_list = [data_dir + '/' + 'target' + '/' + i for i in label_list]

        self.train_list = train_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        train_image = Image.open(self.train_list[idx])
        train_image = self.transform(train_image)

        label_image = Image.open(self.label_list[idx])
        label_image = self.transform(label_image)
        return train_image, label_image


def make_loaders(data_path):
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((128, 128)),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                ])
    train_path = os.path.join(data_path, 'train')
    valid_path = os.path.join(data_path, 'valid')
    test_path = os.path.join(data_path, 'test')
    train_dataset = ImageProcessDataset(train_path, transform=trans)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16)
    valid_dataset = ImageProcessDataset(valid_path, transform=trans)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=16)
    test_dataset = ImageProcessDataset(test_path, transform=trans)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=16)

    return train_loader, valid_loader, test_loader

def make_model(pretrained_model):
    
    if pretrained_model == '' or pretrained_model == None:
        model = ImageProcessingTransformer(
            patch_size=4, depth=6, num_heads=4, ffn_ratio=4, qkv_bias=True,drop_rate=0.2, attn_drop_rate=0.2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), ).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) 
    
    else:
        model = ImageProcessingTransformer(
            patch_size=4, depth=6, num_heads=4, ffn_ratio=4, qkv_bias=True,drop_rate=0.2, attn_drop_rate=0.2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), ).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) 

        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

train_loader, valid_loader, test_loader = make_loaders(data_path)
model, optimizer = make_model(pretrained_model)

if pretrained_model == '' or pretrained_model == None:
    model_name = 'raw_model'
else:
    model_name = pretrained_model

def train_main():
    
    criterion = nn.MSELoss()  

    model.train()
    model.set_task(5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001) 

    # nb_epochs = 300
    for epoch in range(nb_epochs + 1):
        for batch_idx, samples in enumerate(train_loader):

            x_train, y_train = samples
            x_train = x_train.cuda()
            x_train.requires_grad_()
            y_train = y_train.cuda()
            # H(x) 계산
            prediction = model(x_train)
            cost = criterion(prediction, y_train)
            # cost로 H(x) 계산
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, batch_idx+1, len(train_loader),
                cost.item()
                ))
        
        # validation part
        for batch_idx, samples in enumerate(valid_loader):
            with torch.no_grad():
                model.eval()
                x_train, y_train = samples
                x_train = x_train.cuda()
                y_train = y_train.cuda()
                prediction = model(x_train)
                valid_cost = criterion(prediction, y_train)
                print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
                    epoch, nb_epochs, batch_idx+1, len(valid_loader),valid_cost.item()))
            
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': cost,
                    'epoch': epoch,
                    }, model_name)

def test_main():
    model.set_task(5)
    criterion = nn.MSELoss()  

    avg_cost = []
    for batch_idx, samples in enumerate(test_loader):
        with torch.no_grad():
            model.eval()
            x_train, y_train = samples
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            prediction = model(x_train)
            test_cost = criterion(prediction, y_train)
            avg_cost.append(float(test_cost.item()))
            print('Batch {}/{} Cost: {:.6f}'.format(
                batch_idx+1, len(test_loader),test_cost.item()))
    return np.mean(avg_cost)

def main():
    train_main()
    test_result = test_main()
    print(test_result)
    return test_result

main()