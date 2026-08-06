











import numpy as np
import torch
import math

#############################################################################################################
#####                                               #####
############################################################################################################
def get_3DGauss(model_size, s=0, e=15, sigma=1.5, mu=7.5):
    x, y = np.mgrid[s:e:model_size*1j, s:e:model_size*1j]
    z = (1 / (2 * math.pi * sigma ** 2)) * np.exp(-((x - mu) ** 2 + (y - mu) ** 2) / (2 * sigma ** 2))
    z = torch.from_numpy(MaxMinNormalization(z.astype(np.float32)))
    for j in range(model_size):
        for i in range(model_size):
            if z[i, j] < 0.1:
                z[i, j] = 0
    return z

#############################################################################################################
#####                                               #####
############################################################################################################
def MaxMinNormalization(loss_img):
    maxvalue = np.max(loss_img)
    minvalue = np.min(loss_img)
    img = (loss_img - minvalue) / (maxvalue - minvalue)
    img = np.around(img, decimals=2)
    return img
#############################################################################################################
#####                                               #####
############################################################################################################