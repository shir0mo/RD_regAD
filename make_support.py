import os
import random
import argparse
import time
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import os
from PIL import Image
import random
import math
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

## valid
class FSAD_Dataset_fewshot(Dataset):
    def __init__(self, dataset_path, class_name, class_list, is_train=True, resize=256, shot=2):
        assert class_name in class_list, 'class_name: {}, should be in {}'.format(class_name, class_list)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.shot = shot

        # load dataset(対象カテゴリの正常画像のみ)
        self.support_dir = self.load_dataset_folder()

        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize(resize, Image.LANCZOS),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = transforms.Compose(
            [transforms.Resize(resize, Image.NEAREST),
             transforms.ToTensor()])

    def __getitem__(self, idx):

        # support_img の格納
        support_one = self.support_dir[idx]
        support_img = []

        for k in range(self.shot):
            support_img_one = Image.open(support_one[k]).convert('RGB')
            support_img_one = self.transform_x(support_img_one)
            support_img.append(support_img_one)

        return support_img

    def __len__(self):
        return len(self.support_dir)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        support_dir = []
        data_img = {}
        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            data_img[img_type] = []
            img_type_dir = os.path.join(img_dir, img_type)
            img_num = sorted(os.listdir(img_type_dir))
            for img_one in img_num:
                img_dir_one = os.path.join(img_type_dir, img_one)
                data_img[img_type].append(img_dir_one)
        img_dir_train = os.path.join(self.dataset_path, self.class_name, 'train', 'good')
        img_num = sorted(os.listdir(img_dir_train))

        data_train = []
        for img_one in img_num:
            img_dir_one = os.path.join(img_dir_train, img_one)
            data_train.append(img_dir_one)

        support_dir = []
        for img_type in data_img.keys():
            for image_dir_one in data_img[img_type]:
                support_dir_one = []
                for k in range(self.shot):
                    random_choose = random.randint(0, (len(data_train) - 1))
                    support_dir_one.append(data_train[random_choose])
                support_dir.append(support_dir_one)

        return support_dir

def main(_class_, class_list, data_type, shot):

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    if data_type == 'mvtec':
        data_path = '../mvtec/'
    if data_type == 'mpdd':
        data_path = '../mpdd/'
    
    output_dir = '../support_set/' + _class_
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ## 画像を入れるリスト
    img_list = []

    fewshot_dataset = FSAD_Dataset_fewshot(data_path, class_name=_class_, class_list=class_list, is_train=False, resize=224, shot=shot )
    fewshot_loader = torch.utils.data.DataLoader(fewshot_dataset, batch_size=1, shuffle=True, **kwargs)

    for i in range(10):
        Iter = iter(fewshot_loader)
        few_support_img = next(Iter)

        support_img = few_support_img[0]

        for shot_num in range(1, len(few_support_img)):
            support_img = torch.cat([support_img, few_support_img[shot_num]], dim=0)

        img_list.append(support_img)
        
    torch.save(img_list, f'{output_dir}/{shot}_10.pt')
    print('{} was saved.'.format(_class_))
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='mvtec', help='input data type, mvtec or mpdd')
    parser.add_argument('--shot', type=int, default=2, help='shot count')
    args = parser.parse_args() 
    
    if args.data_type == 'mvtec':
        item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                     'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
    if args.data_type == 'mpdd':
        item_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']

    for i in range(len(item_list)):
        main(item_list[i], item_list, args.data_type, args.shot)