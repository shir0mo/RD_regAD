import os
import datetime
import pytz

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import matplotlib

import numpy as np
import torch
import torch.nn.functional as F
import kornia as K

from skimage import morphology
from skimage.segmentation import mark_boundaries

# create log
def create_log_file(_class_):

    # get time
    dt_now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    get_day = str(dt_now.year) + "-" + str(dt_now.month) + "-" + str(dt_now.day)
    get_time = str(dt_now.hour) + "-"  + str(dt_now.minute)

    filename = "../train_log/" + _class_ + "_" + get_day + ".txt"

    with open(filename, "w") as f:
        f.close()
        
    print("create {}".format(filename[3:]))
    
    return filename
    
# print function
def log_and_print(message, file="log.txt"):
    print(message)  # 標準出力

    assert os.path.exists(file), "{} is not exists.".format(file)

    with open(file, "a") as f:
        f.write(message + "\n")
        f.close()

# t-sne visualization
def plot_tsne(features, labels, epoch, cls_list, _class_, _path_):

    tsne_path = "../visualize/" + _class_
    if not os.path.exists(tsne_path):
        os.mkdir(tsne_path)
    else:
        print("directory is already exists.")
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    colors = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#800000"
    ]

    # mpdd_colors = [
    #     "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231"
    # ]

    custom_cmap = mcolors.ListedColormap(colors)

    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=custom_cmap, s=10)

    patches = [mpatches.Patch(color=custom_cmap(i), label=cls_list[i]) for i in range(len(cls_list))]
    plt.legend(handles=patches, title='Classes', bbox_to_anchor=(1., 1), loc='upper left')

    #plt.colorbar(scatter)
    plt.title(f't-SNE Visualization at Epoch {epoch}')
    plt.savefig(f'{tsne_path}/{_path_}_tsne_epoch_{epoch}.png')  # Save the plot for each epoch

    del(tsne_results)

# concat outputs
def embedding_concat(x, y, use_cuda):
    device = torch.device('cuda' if use_cuda else 'cpu')
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2).to(device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z

# calc mahalanobis
def mahalanobis_torch(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)

def denormalization(x):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    # x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x

def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 3, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        #ax_img[3].imshow(mask, cmap='gray')
        #ax_img[3].title.set_text('Predicted mask')
        #ax_img[4].imshow(vis_img)
        #ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()