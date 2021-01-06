import torch
from torchvision import transforms as trans
from torch.utils.data.dataloader import Dataset, DataLoader
import cv2
from PIL import Image
from tqdm import tqdm

class TestingImages(Dataset):
    def __init__(self, transform):
        with open('dataset/testing.txt') as f:
            self.data_list = f.readlines()
        self.data_num = len(self.data_list)
        self.transform = transform
        self.scale = 3

    def __getitem__(self, item):
        filename = 'dataset/testing_lr_images/' + self.data_list[item][:-1]
        img_hr = Image.open(filename)
        img_lr = self.transform(img_hr)
        return img_lr, self.data_list[item][:-1]

    def __len__(self):
        return self.data_num

model = torch.load('checkpoints/model_0035.pth').cuda()

tt = trans.Compose([
    trans.ToTensor(),
])
test_dataset = TestingImages(tt)
loader = DataLoader(test_dataset)

for data in tqdm(loader):
    img_lr, file_name = data
    with torch.no_grad():
        pred = model(img_lr.cuda(), 3)
    img = (pred[0].clip(0, 1)*255).detach().cpu().type(torch.uint8).numpy()[::-1].transpose((1, 2, 0))
    cv2.imwrite('dataset/output/' + file_name[0], img)
