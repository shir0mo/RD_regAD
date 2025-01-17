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
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
from tqdm import tqdm

from random import sample
from utils import create_log_file, log_and_print, plot_tsne, embedding_concat, mahalanobis_torch
import time

import os

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

# train時の評価にも利用
def evaluation(encoder, pred, dataloader, device, shot, _class_=None):

    #Hyper params
    image_size = 224
    idx = torch.tensor(sample(range(0, 448), 100))

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    test_path = '../mvtec/' + _class_
    # support data
    print('Loading Fixed Support Set')
    fixed_fewshot_list = torch.load(f'../CAReg/support_set/{_class_}/{shot}_10.pt')
    
    fewshot_data = SupportDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, shot=shot)
    fewshot_dataloader = torch.utils.data.DataLoader(fewshot_data, batch_size=1,  num_workers=1, shuffle=True)
    
    # 評価モード
    encoder.eval()
    pred.eval()

    src_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    tgt_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    image_auc_list = []
    pixel_auc_list = []
    
    time_list = []

    for i in tqdm(range(10)):

        # support img
        support_img = fixed_fewshot_list[i]
        support_img = augment_support_data(support_img)

        # meausure time
        start_time = time.perf_counter()

        with torch.no_grad():
            support_img = support_img.to(device)
            inputs = encoder(support_img)
            feat, z_support, x = pred(inputs)
            # outputs = decoder(btl)
        feat_s = torch.mean(feat, dim=0, keepdim=True)
        src_outputs['layer1'].append(encoder.enc1_output)
        src_outputs['layer2'].append(encoder.enc2_output)
        src_outputs['layer3'].append(encoder.enc3_output)

        
        for k, v in train_outputs.items():
            src_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        embedding_vectors = src_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, src_outputs[layer_name], True)
        
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

        with torch.no_grad():
            img_num = 0
            for query_img, mask, y in dataloader:
                query_imgs.extend(query_img.cpu().detach().numpy())
                gt_list.extend(y.cpu().detach().numpy())
                mask_list.extend(mask.cpu().detach().numpy())

                # prediction
                query_img = query_img.to(device)
                _inputs = encoder(query_img)
                _feat, z_test, x_test = pred(_inputs)

                tgt_outputs['layer1'].append(encoder.enc1_output)
                tgt_outputs['layer2'].append(encoder.enc2_output)
                tgt_outputs['layer3'].append(encoder.enc3_output)
                
                img_num += 1
        
        for k, v in test_outputs.items():
            tgt_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, tgt_outputs[layer_name], True)
        
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
        scores = np.asarray(scores_list)
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
        gt_mask = np.asarray(gt_mask_list)
        gt_mask = (gt_mask > 0.5).astype(np.int_)
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        pixel_auc_list.append(per_pixel_rocauc)
    
    image_auc_list = np.array(image_auc_list)
    pixel_auc_list = np.array(pixel_auc_list)
    mean_img_auc = np.mean(image_auc_list, axis = 0)
    mean_pixel_auc = np.mean(pixel_auc_list, axis = 0)

    return mean_img_auc, mean_img_auc, sum(time_list) / len(time_list)

def test(_class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(_class_)

    num_class = 14
    # experiments settings
    shot = 4

    data_transform, gt_transform = get_data_transforms(224, 224)
    test_path = '../mvtec/' + _class_
    ckp_path = '../checkpoints/' + 'res18_' + _class_ + '_2025-1-8_15-15.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, bn = resnet18(num_class, pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    # auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device,_class_)
    auroc_sp, inference_time = evaluation(encoder, bn, decoder, test_dataloader, device, shot, _class_)
    print('Image level AUCROC: {:.3f}, time: {:.3f}[s]'
                      .format(auroc_sp, inference_time))
    return auroc_sp

import os

def visualization(_class_):
    print(_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = '../mvtec/' + _class_
    ckp_path = './checkpoints/' + 'rm_1105_wres50_ff_mm_'+_class_+'.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)

    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    count = 0
    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            if (label.item() == 0):
                continue
            #if count <= 10:
            #    count += 1
            #    continue

            decoder.eval()
            bn.eval()

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            #inputs.append(feature)
            #inputs.append(outputs)
            #t_sne(inputs)


            anomaly_map, amap_list = cal_anomaly_map([inputs[-1]], [outputs[-1]], img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            #if not os.path.exists('./results_all/'+_class_):
            #    os.makedirs('./results_all/'+_class_)
            #cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'org.png',img)
            #plt.imshow(img)
            #plt.axis('off')
            #plt.savefig('org.png')
            #plt.show()
            ano_map = show_cam_on_image(img, ano_map)
            #cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'ad.png', ano_map)
            plt.imshow(ano_map)
            plt.axis('off')
            #plt.savefig('ad.png')
            plt.show()

            gt = gt.cpu().numpy().astype(int)[0][0]*255
            #cv2.imwrite('./results/'+_class_+'_'+str(count)+'_'+'gt.png', gt)

            #b, c, h, w = inputs[2].shape
            #t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            #print(c.shape)
            #t_sne([t_feat, s_feat], c)
            #assert 1 == 2

            #name = 0
            #for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
                #ano_map = show_cam_on_image(img, ano_map)
                #cv2.imwrite(str(name) + '.png', ano_map)
                #plt.imshow(ano_map)
                #plt.axis('off')
                #plt.savefig(str(name) + '.png')
                #plt.show()
            #    name+=1
            count += 1
            #if count>20:
            #    return 0
                #assert 1==2


def vis_nd(name, _class_):
    print(name,':',_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    ckp_path = './checkpoints/' + name + '_' + str(_class_) + '.pth'
    train_dataloader, test_dataloader = load_data(name, _class_, batch_size=16)

    encoder, bn = resnet18(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)

    ckp = torch.load(ckp_path)

    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    decoder.eval()
    bn.eval()

    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []

    count = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            #if count <= 10:
            #    count += 1
            #    continue
            img = img.to(device)
            inputs = encoder(img)
            #print(inputs[-1].shape)
            outputs = decoder(bn(inputs))


            anomaly_map, amap_list = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            #anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'org.png',img)
            #plt.imshow(img)
            #plt.axis('off')
            #plt.savefig('org.png')
            #plt.show()
            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'ad.png', ano_map)
            #plt.imshow(ano_map)
            #plt.axis('off')
            #plt.savefig('ad.png')
            #plt.show()

            #gt = gt.cpu().numpy().astype(int)[0][0]*255
            #cv2.imwrite('./results/'+_class_+'_'+str(count)+'_'+'gt.png', gt)

            #b, c, h, w = inputs[2].shape
            #t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            #print(c.shape)
            #t_sne([t_feat, s_feat], c)
            #assert 1 == 2

            #name = 0
            #for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
                #ano_map = show_cam_on_image(img, ano_map)
                #cv2.imwrite(str(name) + '.png', ano_map)
                #plt.imshow(ano_map)
                #plt.axis('off')
                #plt.savefig(str(name) + '.png')
                #plt.show()
            #    name+=1
            #count += 1
            #if count>40:
            #    return 0
                #assert 1==2
            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))  # np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        ano_score = (prmean_list_sp-np.min(prmean_list_sp))/(np.max(prmean_list_sp)-np.min(prmean_list_sp))
        vis_data = {}
        vis_data['Anomaly Score'] = ano_score
        vis_data['Ground Truth'] = np.array(gt_list_sp)
        #print(type(vis_data))
        #np.save('vis.npy',vis_data)
        with open('vis.pkl','wb') as f:
            pickle.dump(vis_data,f,pickle.HIGHEST_PROTOCOL)


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

# def detection(encoder, bn, decoder, dataloader,device,_class_):
#     #_, t_bn = resnet50(pretrained=True)
#     bn.load_state_dict(bn.state_dict())
#     bn.eval()
#     #t_bn.to(device)
#     #t_bn.load_state_dict(bn.state_dict())
#     decoder.eval()
#     gt_list_sp = []
#     prmax_list_sp = []
#     prmean_list_sp = []
#     with torch.no_grad():
#         for img, label in dataloader:

#             img = img.to(device)
#             if img.shape[1] == 1:
#                 img = img.repeat(1, 3, 1, 1)
#             label = label.to(device)
#             inputs = encoder(img)
#             outputs = decoder(bn(inputs))
#             anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], 'acc')
#             anomaly_map = gaussian_filter(anomaly_map, sigma=4)


#             gt_list_sp.extend(label.cpu().data.numpy())
#             prmax_list_sp.append(np.max(anomaly_map))
#             prmean_list_sp.append(np.sum(anomaly_map))#np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

#         gt_list_sp = np.array(gt_list_sp)
#         indx1 = gt_list_sp == _class_
#         indx2 = gt_list_sp != _class_
#         gt_list_sp[indx1] = 0
#         gt_list_sp[indx2] = 1


#         auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
#         auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)
#     return auroc_sp_max, auroc_sp_mean

if __name__ == '__main__':
    
    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
             'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']

    test('tile')
    # for i in range(len(item_list)):
    #     test(item_list[i])