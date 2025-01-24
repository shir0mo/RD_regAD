import torch
from dataset import get_data_transforms
from augmentation import augment_support_data
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from dataset import MVTecDataset, SupportDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
from tqdm import tqdm
from collections import OrderedDict

from random import sample
from utils import create_log_file, log_and_print, plot_tsne, embedding_concat, mahalanobis_torch, plot_fig
import time

import argparse 

import os


# train時の評価にも利用
def evaluation(encoder, pred, dataloader, device, shot, _class_=None):

    #Hyper params
    image_size = 224
    idx = torch.tensor(sample(range(0, 448), 100))

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    test_path = '../mpdd/' + _class_
    # support data
    print('Loading Fixed Support Set')
    fixed_fewshot_list = torch.load(f'../support_set/{_class_}/{shot}_10.pt')
    
    fewshot_data = SupportDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, shot=shot)
    fewshot_dataloader = torch.utils.data.DataLoader(fewshot_data, batch_size=1,  num_workers=1, shuffle=True)
    
    # 評価モード
    encoder.eval()
    pred.eval()

    image_auc_list = []
    pixel_auc_list = []
    time_list = []

    for i in tqdm(range(1)):

        src_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        tgt_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # support img
        support_img = fixed_fewshot_list[i]
        support_img = augment_support_data(support_img)

        # meausure time
        with torch.no_grad():
            support_img = support_img.to(device)
            inputs = encoder(support_img)
            #feat, z_support, x = pred(inputs)
            
        #feat_s = torch.mean(feat, dim=0, keepdim=True)
        src_outputs['layer1'].append(encoder.enc1_output.detach())
        src_outputs['layer2'].append(encoder.enc2_output.detach())
        src_outputs['layer3'].append(encoder.enc3_output.detach())

        for k, v in src_outputs.items():
            src_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        patch_lib = []
        embedding_vectors = src_outputs['layer2']
        ploxy_map_size = embedding_vectors.shape[-2:]
        print(ploxy_map_size)

        resize = torch.nn.AdaptiveAvgPool2d(ploxy_map_size)

        resized_maps = [resize(src_outputs[layer_name]) for layer_name in ['layer1', 'layer3']]
        patch_lib.append(torch.cat(resized_maps, 1))

        embedding_vectors = torch.cat(patch_lib, 0)
        
        # embedding_vectors = src_outputs['layer1']
        # for layer_name in ['layer2', 'layer3']:
        #     embedding_vectors = embedding_concat(embedding_vectors, src_outputs[layer_name], True)
        
        # randomly select d dimension
        #embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors, dim=0)
        cov = torch.zeros(C, C, H * W).to(device)
        I = torch.eye(C).to(device)
        for i in range(H * W):
            cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
        train_outputs = [mean, cov]
        
        # torch version
        query_imgs = []
        gt_list = []
        mask_list = []
        
        start_time = time.perf_counter()
        with torch.no_grad():
            img_num = 0
            for query_img, mask, y in dataloader:
                query_imgs.extend(query_img.cpu().detach().numpy())
                gt_list.extend(y.cpu().detach().numpy())
                mask_list.extend(mask.cpu().detach().numpy())

                # prediction
                query_img = query_img.to(device)
                _inputs = encoder(query_img)
                #_feat, z_test, x_test = pred(_inputs)

                tgt_outputs['layer1'].append(encoder.enc1_output)
                tgt_outputs['layer2'].append(encoder.enc2_output)
                tgt_outputs['layer3'].append(encoder.enc3_output)
                
                img_num += 1
        
        for k, v in tgt_outputs.items():
            tgt_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = tgt_outputs['layer2']
        ploxy_map_size = embedding_vectors.shape[-2:]

        resize = torch.nn.AdaptiveAvgPool2d(ploxy_map_size)

        resized_maps = [resize(tgt_outputs[layer_name]) for layer_name in ['layer1', 'layer3']]
        embedding_vectors = torch.cat(resized_maps, 1)
        # for layer_name in ['layer1', 'layer3']:
        #     embedding_vectors = embedding_concat(embedding_vectors, tgt_outputs[layer_name], True)
        
        # randomly select d dimension
        #embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        dist_list = []

        for i in range(H * W):
            mean = train_outputs[0][:, i]
            conv_inv = torch.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis_torch(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = torch.tensor(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        score_map = F.interpolate(dist_list.unsqueeze(1), size=query_img.size(2), mode='bilinear',
                              align_corners=False).squeeze().numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # meausure time
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        time_list.append(elapsed_time / img_num)

        # return score_map, query_imgs, gt_list, mask_list
        scores = np.asarray(score_map)
        # Normalization
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        image_auc_list.append(img_roc_auc)

        # calculate per-pixel level ROCAUC
        gt_mask = np.asarray(mask_list)
        gt_mask = (gt_mask > 0.5).astype(np.int_)
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        pixel_auc_list.append(per_pixel_rocauc)

        # get optimal threshold
        gt_mask = np.asarray(mask_list)
        gt_mask = (gt_mask > 0.5).astype(np.int_)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        print(f1)

        save_dir = '../test_img/' + _class_ + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plot_fig(query_imgs, scores, mask_list, threshold, save_dir, _class_)
    
    image_auc_list = np.array(image_auc_list)
    pixel_auc_list = np.array(pixel_auc_list)
    mean_img_auc = np.mean(image_auc_list, axis = 0)
    mean_pixel_auc = np.mean(pixel_auc_list, axis = 0)

    return mean_img_auc, mean_pixel_auc, sum(time_list) / len(time_list)

def test(_class_, item_list, data_name, _date_, shot):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(_class_)

    num_class = len(item_list) - 1
    image_size = 224
    # experiments settings
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    test_path = '../' + data_name + '/'
    ckp_path = '../checkpoints/' + 'res18_' + _class_ + '_' + _date_ + '.pth'

    # dataset
    test_data = MVTecDataset(test_path, class_name=_class_, is_train=False, resize=image_size, shot=shot)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, pred = resnet18(num_class, pretrained=True)

    test_data = MVTecDataset(test_path, class_name=_class_, is_train=False, resize=image_size, shot=shot)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    encoder, pred = resnet18(num_class, pretrained=True)
    ckp = torch.load(ckp_path)
    
    # for k, v in list(ckp['pred'].items()):
    #     if 'memory' in k:
    #         ckp['pred'].pop(k)
    
    encoder.load_state_dict(ckp['encoder'])
    pred.load_state_dict(ckp['pred'])

    encoder = encoder.to(device)
    pred = encoder.to(device)
    # auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device,_class_)
    # eval
    
    auroc_sp, auroc_px, inference_time = evaluation(encoder, pred, test_dataloader, device, shot, _class_)
    print('Image level AUCROC: {:.3f}, Image level AUCROC: {:.3f}, time: {:.3f}[s]'
                  .format(auroc_sp, auroc_px, inference_time))
    return auroc_sp
    
if __name__ == '__main__':
    
    #setup_seed(111)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='mvtec', help='input data type, mvtec or mpdd')
    parser.add_argument('--date', default='1-21', help='checkpoints date')
    parser.add_argument('--shot', default=2, help='shot')
    
    args = parser.parse_args() 
    if args.data_type == 'mvtec':
        item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                     'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
    if args.data_type == 'mpdd':
        item_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    
    # sbatch用
    if os.getcwd() == '/k_home/g518nabe/rd_base':
        os.chdir('./RD_RegAD')
        print(os.getcwd())
        
    # for i in range(6,12):
    #     train(item_list[i], item_list, args.data_type)

    test(item_list[1], item_list, args.data_type, args.date, int(args.shot))