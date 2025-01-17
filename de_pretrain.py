# This is a sample Python script.


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset, make_train_data, ClassificationDataset, SupportDataset
import torch.backends.cudnn as cudnn
import argparse
from torch.nn import functional as F
from loss import center_loss_func, update_center, loss_fucntion, loss_concat
from utils import create_log_file, log_and_print, plot_tsne

from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

import time
from tqdm import tqdm

import argparse 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []

    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
            
    return anomaly_map, a_map_list

# train時の評価にも利用
def evaluation(encoder, bn, decoder, dataloader, device, _class_=None):

    #Hyper params
    image_size = 256

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    test_path = '../mvtec/' + _class_
    
    # 評価モード
    encoder.eval()
    bn.eval()
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    
    auc_list = []
    time_list = []

    for i in tqdm(range(10)): 
        with torch.no_grad():
            for img, gt, label, _ in dataloader:

                img = img.to(device)
                inputs = encoder(img)
                btl, _, _x = bn(inputs)

                outputs = decoder(btl)
                
                # segmentation
                anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0

                gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
                pr_list_px.extend(anomaly_map.ravel())

                gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
                pr_list_sp.append(np.max(anomaly_map))
            #ano_score = (pr_list_sp - np.min(pr_list_sp)) / (np.max(pr_list_sp) - np.min(pr_list_sp))
            #vis_data = {}
            #vis_data['Anomaly Score'] = ano_score
            #vis_data['Ground Truth'] = np.array(gt_list_sp)
            # print(type(vis_data))
            # np.save('vis.npy',vis_data)
            #with open('{}_vis.pkl'.format(_class_), 'wb') as f:
            #    pickle.dump(vis_data, f, pickle.HIGHEST_PROTOCOL)


        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)

    return auroc_px, auroc_sp

def train(_class_, item_list, data_name):

    # copy
    class_list = []
    for i in range(len(item_list)):
        # 対象カテゴリ以外を省く
        if item_list[i] != _class_:
           class_list.append(item_list[i])

    print(_class_)
    print(class_list)
    torch.cuda.reset_max_memory_allocated()

    # Hyper params:
    epochs = 50
    warmup_epochs = 10
    learning_rate = 0.005
    batch_size = 16
    image_size = 256
    num_class = len(class_list)

    center_alpha = 1.0
    center_beta = 1.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # train dataframe
    train_df = make_train_data(_class_)
    # create log
    log_path = create_log_file(_class_)
 
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    test_path = '../' + data_name + '/' + _class_
    ckp_path = '../decoder/' + 'res18_dec_'+ _class_ + '.pth'

    # dataset
    train_dataset = ClassificationDataset(train_df)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=1, shuffle=False)

    # encoder: freeze
    encoder, bn = resnet18(num_class, pretrained=True)
    encoder = encoder.to(device)

    # bn, decoder: train
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)

    # print model
    # print(bn)
    optimizer = torch.optim.Adam(list(bn.parameters())+list(decoder.parameters()), lr=learning_rate, betas=(0.5,0.999))

    # compare scores 
    auc_old, auc_pre = 0.000, 0.000

    for epoch in range(epochs):
    
        bn.train()
        decoder.train()
        loss_list = []

        # watching loss
        cosloss_list = []
        centerloss_list = []

        for img, label in tqdm(train_dataloader):
            img = img.to(device)
            label = label.to(device, dtype=torch.int64)
            inputs = encoder(img)
            btl, z, x = bn(inputs)
            outputs = decoder(btl)

                       # 損失計算
            cosloss = loss_fucntion(inputs, outputs)
            centerloss = F.cross_entropy(x, label)  + center_alpha * center_loss_func(bn, z, label)
            loss = cosloss + centerloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_center(bn, label, center_beta, num_class)
            cosloss_list.append(cosloss.item())
            centerloss_list.append(centerloss.item())
            loss_list.append(loss.item())

            #ip1_loader.append(z)
            # idx_loader.append((label))

        # feat = torch.cat(ip1_loader, 0).cpu()
        # labels_list = torch.cat(idx_loader, 0).cpu()
        # plot_tsne(feat.detach().numpy(), labels_list.detach().numpy(), epoch, class_list, _class_, log_path[13:-4])
        
        log_and_print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)), log_path)
        
        # eval
        auroc_px, auroc_sp = evaluation(encoder, bn, decoder, test_dataloader, device, _class_)
        log_and_print('Image level AUCROC: {:.3f}, Pixel level AUCROC: {:.3f}'
                      .format(auroc_sp, auroc_px), log_path)
        
        auc_pre = auroc_sp + auroc_px
        if auc_old <= auc_pre:
            auc_old = auc_pre
            torch.save({'decoder': decoder.state_dict()}, ckp_path)
            print("saveed {} .".format(ckp_path[3:]))
    # return auroc_px, auroc_sp, aupro_px


if __name__ == '__main__':

    setup_seed(111)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='mvtec', help='input data type, mvtec or mpdd')
    
    args = parser.parse_args() 
    if args.data_type == 'mvtec':
        item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                     'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
    if args.data_type == 'mpdd':
        item_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']

    # sbatch用
    if os.getcwd() == '/k_home/g518nabe/rd_base':
        os.chdir('./RDbc')
        print(os.getcwd())
        
    for i in range(len(item_list)):
        train(item_list[i], item_list, args.data_type)