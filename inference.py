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
parser.add_argument('--inf_dir', type=str, default='test')
parser.add_argument('--pretrained_model', type=str, default='model_1016_0.2')

args = parser.parse_args()
data_path = args.data_path
pretrained_model = args.pretrained_model
inf_dir = args.inf_dir

class ImageProcessDataset(Dataset):
  def __init__(self, data_dir, transform):

    train_path = data_dir + '/' + 'hazy'

    try:
      train_list = sorted(os.listdir(train_path))
    except:
      raise ValueError
    
    train_list = [data_dir + '/' + 'hazy' + '/' + i for i in train_list]

    self.train_list = train_list
    self.transform = transform

  def __len__(self):
    return len(self.train_list)
  
  def __getitem__(self, idx):
    train_image = Image.open(self.train_list[idx])
    train_image = self.transform(train_image)

    return train_image, self.train_list[idx]

def make_loaders(data_path,inf_dir):
    trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((128,128)),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                    ])
    
    test_path = os.path.join(data_path, inf_dir)
    test_dataset = ImageProcessDataset(test_path, transform=trans)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)

    return test_loader

inf_loader = make_loaders(data_path, inf_dir) # no target data

def main():
    model.set_task(5)
    criterion = nn.MSELoss()  
    for batch_idx, samples in enumerate(inf_loader):
        with torch.no_grad():
            model.eval()
            x_train, img_path = samples
            x_train = x_train.cuda()
            prediction = model(x_train)
            im = Image.fromarray(np.transpose(prediction[0].cpu().detach().numpy() * 0.5 + 0.5, (1,2,0)))
            im_name = str(img_path[0]).split('.')[0]
            im.save(im_name + '_dehazed' + '.' + 'jpg')
