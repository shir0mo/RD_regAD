from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import random
import warnings
warnings.filterwarnings("ignore")

CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper'
]

def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        #transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

# Few-shot query Dataset
class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path='../data/mvtec_anomaly_detection',
                 class_name='bottle',
                 is_train=True,
                 resize=256,
                 shot=2
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.shot = shot
        # load dataset
        self.query_dir, self.support_dir, self.query_mask = self.load_dataset_folder()
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
        query_one, support_one, mask_one = self.query_dir[idx], self.support_dir[idx], self.query_mask[idx]
        query_img = Image.open(query_one).convert('RGB')
        query_img = self.transform_x(query_img)

        # support_img = []
        # for k in range(self.shot):
        #     support_img_one = Image.open(support_one[k]).convert('RGB')
        #     support_img_one = self.transform_x(support_img_one)
        #     support_img.append(support_img_one)

        if 'good' in mask_one:
            mask = torch.zeros([1, self.resize, self.resize])
            y = 0
        else:
            mask = Image.open(mask_one)
            mask = self.transform_mask(mask)
            y = 1
        
        return query_img, mask, y

    def __len__(self):
        return len(self.query_dir)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        query_dir, support_dir = [], []
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

        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        query_dir, support_dir, query_mask = [], [], []
        for img_type in data_img.keys():
            for image_dir_one in data_img[img_type]:
                support_dir_one = []
                query_dir.append(image_dir_one)
                query_mask_dir = image_dir_one.replace('test', 'ground_truth')
                query_mask_dir = query_mask_dir[:-4] + '_mask.png'
                query_mask.append(query_mask_dir)
                for k in range(self.shot):
                    random_choose = random.randint(0, (len(data_train) - 1))
                    support_dir_one.append(data_train[random_choose])
                support_dir.append(support_dir_one)

        assert len(query_dir) == len(support_dir) == len(
            query_mask), 'number of query_dir and support_dir should be same'
        return query_dir, support_dir, query_mask

# Few-shot support Dataset
class SupportDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, shot):

        # add train path
        self.img_path = os.path.join(root, 'train', 'good')

        self.transform = transform
        self.gt_transform = gt_transform
        self.shot = shot

        # load dataset
        self.img_paths, self.gt_paths= self.load_dataset()

    def __getitem__(self, idx):
        img_path, gt = self.img_paths[idx], self.gt_paths[idx]

        support_img = []
        support_gt = []

        for k in range(self.shot):
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
            support_img.append(img)
            support_gt.append(gt)

            assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
            
        return support_img, support_gt
    
    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []

        img_paths = glob.glob(os.path.join(self.img_path) + "/*.png")
        img_tot_paths.extend(img_paths)
        gt_tot_paths.extend([0] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths

    def __len__(self):
        return len(self.img_paths)

# train
def make_train_data(exclude_label):
    # データセットのパスを設定
    output_path = '../mvtec_train'

    # ラベル情報を読み込む
    labels_df = pd.read_csv(os.path.join(output_path, 'labels.csv'))

    # 特定のラベルを除外
    filtered_df = labels_df[labels_df['label'] != exclude_label]

    # ラベルをエンコード
    label_encoder = LabelEncoder()
    filtered_df.loc[:, 'label'] = label_encoder.fit_transform(filtered_df['label'])
    # 訓練セットとテストセットに分割
    return filtered_df

# 分類タスク用
class ClassificationDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

        if transform == None:
            mean_train = [0.485, 0.456, 0.406]
            std_train = [0.229, 0.224, 0.225]

            self.transform = transforms.Compose([
                transforms.Resize(224, Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_train, std=std_train)
            ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label