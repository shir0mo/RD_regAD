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
from dataset import MVTecDataset, make_train_data, ClassificationDataset, SupportDataset

import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test
from torch.nn import functional as F
from loss import center_loss_func, update_center, loss_fucntion, loss_concat
from utils import create_log_file, log_and_print, plot_tsne, embedding_concat, mahalanobis_torch
from collections import OrderedDict
from stn import stn_net

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

    learning_rate = 0.005
    momentum = 0.9
    batch_size = 32
    image_size = 224

    num_class = len(class_list)
    center_lambda = 0.5
    center_alpha = 1.0

    # experiments settings
    shot = 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # train dataframe
    train_df = make_train_data(_class_)
    # create log
    #log_path = create_log_file(_class_)
    log_path = "./test.txt"
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    test_path = '../' + data_name + '/'
    ckp_path = '../checkpoints/' + 'res18_'+ _class_ + '.pth'

    # dataset
    train_dataset = ClassificationDataset(train_df)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    
    test_data = MVTecDataset(test_path, class_name=_class_, is_train=False, resize=image_size, shot=shot)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, pred = resnet18(num_class, pretrained=True)

    args.stn_mode = 'affine'
    stn = stn_net(args)
    stn = stn.to(device)
    # encoder = encoder.to(device)
    pred = pred.to(device)
    # encoder.train()

    optimizer = torch.optim.SGD(list(stn.parameters())+list(pred.parameters()), lr=learning_rate, momentum=momentum)

    # compare scores 
    auc_old, auc_pre = 0.000, 0.000

    for epoch in range(epochs):

        # time
        encoder.train()        
        pred.train()

        loss_list = []

        # watching loss
        celoss_list = []
        centerloss_list = []

        # visualize
        ip1_loader = []
        idx_loader = []
        
        for img, label in tqdm(train_loader):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            img = img.to(device)
            label = label.to(device, dtype=torch.int64)
            inputs = stn(img)
            z, x = pred(inputs)

            # 損失計算
            ce_loss = F.cross_entropy(x, label)
            center_loss = center_lambda * 0.5 * center_loss_func(pred, z, label)
            loss = ce_loss + center_loss

            torch.cuda.synchronize()
            end_time = time.perf_counter()
            print(end_time - start_time)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_center(pred, label, center_alpha, num_class)
            celoss_list.append(ce_loss.item())
            centerloss_list.append(center_loss.item())
            loss_list.append(loss.item())
            #ip1_loader.append(z)
        
            #idx_loader.append((label))

        #elapsed_time = end_time - start_time

        feat = torch.cat(ip1_loader, 0).cpu()
        labels_list = torch.cat(idx_loader, 0).cpu()
        plot_tsne(feat.detach().numpy(), labels_list.detach().numpy(), epoch, class_list, _class_, log_path[13:-4])

        log_and_print('ce_loss: {:.4f}, center_loss: {:.4f}'
                      .format(np.mean(celoss_list), np.mean(centerloss_list)), log_path)
        log_and_print('epoch [{}/{}], loss:{:.4f}'
                      .format(epoch + 1, epochs, np.mean(loss_list)), log_path)
        log_and_print("epoch {}, time:{} m {} s"
                      .format(epoch + 1, int(elapsed_time // 60), int(elapsed_time % 60)), log_path)
        
        # eval
        auroc_sp, auroc_px, inference_time = evaluation(encoder, pred, test_dataloader, device, shot, _class_)
        log_and_print('Image level AUCROC: {:.3f}, Image level AUCROC: {:.3f}, time: {:.3f}[s]'
                      .format(auroc_sp, auroc_px, inference_time), log_path)

        auc_pre = auroc_sp

        if auc_old <= auc_pre:
            auc_old = auc_pre
            # auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
            # print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            torch.save({'encoder': encoder.state_dict(),
                        'pred': pred.state_dict()}, ckp_path)
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
        os.chdir('./RD_RegAD')
        print(os.getcwd())
        
    # for i in range(6,12):
    #     train(item_list[i], item_list, args.data_type)
    train('hazelnut', item_list, 'mvtec')