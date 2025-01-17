import torch.nn
from torch.nn import functional as F

# center loss
def center_loss_func(model, feature, target):
    return F.mse_loss(feature, model.centers[target])

# optim center
def update_center(model, target, beta_center, num_class):
    feature = model.x_center.detach()
    delta = torch.zeros(num_class, 512).cuda()
    for i in range(feature.shape[0]):
        delta[target[i]] += feature[i] - model.centers[target[i]]
    for t, c in zip(*torch.unique(target, return_counts=True)):
        delta[t] = delta[t] / (1 + c)
    model.centers += beta_center * delta

# rd loss
def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss