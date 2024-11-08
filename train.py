import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sklearn
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import cv2
from net import CrossViT
from tqdm import tqdm
import time
import csv
import sys

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(torch.cuda.get_device_name(0))
else:
    print("cpu")

img_dirs = np.array(['./image_bloks/0/' + i for i in os.listdir('./image_bloks/0/')] +
                    ['./image_bloks/1/' + i for i in os.listdir('./image_bloks/1/')])
np.random.seed(44)
np.random.shuffle(img_dirs)
img_dirs = img_dirs.tolist()
labels = list(map(lambda x:np.float32(x.split('/')[-2]), list(img_dirs)))
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(img_dirs, labels)
transform_t = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    #A.Rotate(limit=180, p=0.5), 
    #A.ShiftScaleRotate(shift_limit=0.1,
                       #scale_limit=0.15,
                       #rotate_limit=60,
                       #p=0.7),
    #A.HueSaturationValue(
        #hue_shift_limit=30,
        #sat_shift_limit=30,
        #val_shift_limit=20,
        #p=0.7),
    #A.RandomBrightnessContrast(
        #brightness_limit=(-0.2, 0.2),
        #contrast_limit=(-0.2, 0.2),
        #p=0.7),
    A.Resize(48, 48),
    A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.),
    #ToTensorV2(transpose_mask=True, p=1.0, always_apply=None)
])

class Luna16Dataset(Dataset):
    def __init__(self, img_dir_ds, label_ds, transform=None):
        self.img_dir = img_dir_ds
        self.label = label_ds
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img = np.load(self.img_dir[idx]).astype(np.float32)
        img = self.transform(image=img)['image']
        img = torch.from_numpy(img).permute(2,0,1)
        img = img.to(torch.float32)
        label = self.label[idx]
        return img, label, self.img_dir[idx]


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

for i, (train_index, test_index) in enumerate(skf.split(img_dirs, labels)):
    if i<0:
        continue
    print(i, 'fold:')
    train_ds = Luna16Dataset([img_dirs[idx] for idx in train_index], [labels[idx] for idx in train_index], transform=transform_t)
    train_dl = DataLoader(
                train_ds,
                batch_size=32,
                shuffle=True,
                drop_last=False
                )
    with open('./fold%d.csv' % i, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(zip([img_dirs[idx] for idx in test_index], [labels[idx] for idx in test_index]))
    model = CrossViT(pretrained=True).to(device=device)
    opt = optim.AdamW(model.parameters(), lr=0.00002)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.90, last_epoch=-1)
    epoch_num = 120
    loss_list = []
    for epoch in range(epoch_num):
        loss_sum = 0.
        start_time = time.time()
        model.train()
        opt.zero_grad()
        for nums, (imgx, labely, idxdir) in enumerate(tqdm(train_dl)):
            
            imgx = imgx.to(device=device)
            labely = labely.to(device=device)
            output0, output1 = model(imgx)
            outputs = torch.where(torch.abs(output0-labely)>torch.abs(output1-labely), output0, output1)
            loss = F.binary_cross_entropy(outputs, labely, reduction='mean')+F.binary_cross_entropy(output0, labely, reduction='mean')+F.binary_cross_entropy(output1, labely, reduction='mean')
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        scheduler.step()
        print('Epoch:', epoch, 'Train_loss: %f'%(loss_sum/(nums+1)), "Time: %f"%(mtime - start_time))
        loss_list.append(loss_sum/(nums+1))
    np.save('loss.npy', loss_list)
    torch.save(model.state_dict(), './fold%d.pth'%i)
    
        

