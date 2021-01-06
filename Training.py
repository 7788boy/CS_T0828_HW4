import torch
import utility
import model
import loss
from option import args
from trainer import Trainer
from torchvision import transforms as trans
from torch.utils.data.dataloader import Dataset, DataLoader
from PIL import Image
import cv2
import torch.nn.functional as F

class TrainingImages(Dataset):
    def __init__(self, transform):
        with open('training.txt') as f:
            self.data_list = f.readlines()
        self.data_num = len(self.data_list)
        self.transform = transform
        self.scale = 3

    def __getitem__(self, item):
        filename = 'training_hr_images/' + self.data_list[item][:-1]
        img_hr = Image.open(filename)
        img_hr = self.transform(img_hr)
        h, w = list(img_hr.shape[-2:])
        img_size_lr = (h//3, w//3)
        img_size_hr = (img_size_lr[0]*3, img_size_lr[1]*3)
        img_lr = F.interpolate(img_hr[None], size=img_size_lr)[0]
        img_hr = F.interpolate(img_hr[None], size=img_size_hr)[0]
        return img_lr, img_hr, filename, self.scale

    def __len__(self):
        return self.data_num

torch.manual_seed(1)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    tt = trans.Compose([
        trans.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
        trans.RandomHorizontalFlip(),
        trans.RandomVerticalFlip(),
        trans.ToTensor(),
    ])
    d = TrainingImages(tt)
    loader = DataLoader(d)

    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint)
    t = Trainer(args, loader, model, loss, checkpoint)
    counter = 0
    while not t.terminate():
        t.train()
        torch.save(model, 'checkpoints/model_%04d.pth' % counter)
        counter += 1

    checkpoint.done()

