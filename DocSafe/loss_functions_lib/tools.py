

"""
@--19.08.2022--@
Author: github/farhadsh1992
INFO:
LAST_UPDATE:
"""


import os
import shutil
import torch
import numpy as np
import cv2

from FarhadCV.Tools import tcolors

import warnings
warnings.filterwarnings('ignore')


######################################################################################################################
#######                                                                      #######
######################################################################################################################
def lr_schedule(epoch_idx):
    if epoch_idx < 200:
        return 0.001
    elif epoch_idx < 400:
        return 0.0003
    elif epoch_idx < 600:
        return 0.0001
    else:
        return 0.00003

######################################################################################################################
#######                                                                      #######
######################################################################################################################
def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            
            



######################################################################################################################
#######                                                                      #######
######################################################################################################################

def mkdirfile(path):
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
######################################################################################################################
#######                                                                      #######
######################################################################################################################

class Batch_ingular_value_decompositions():
    def __init__(self, precent):
        super(Batch_ingular_value_decompositions, self).__init__()
        self.precent = precent
        
    def call(self, a):
        UR, sigmaR, VR = np.linalg.svd(a[:,:,0], full_matrices=False)
        UG, sigmaG, VG = np.linalg.svd(a[:,:,1], full_matrices=False)
        UB, sigmaB, VB = np.linalg.svd(a[:,:,2], full_matrices=False)
    
        np_a_approx_R = np.dot(UR[:, :self.precent], np.dot(np.diag(sigmaR[:self.precent]), VR[:self.precent, :]))
        np_a_approx_G = np.dot(UG[:, :self.precent], np.dot(np.diag(sigmaG[:self.precent]), VG[:self.precent, :]))
        np_a_approx_B = np.dot(UB[:, :self.precent], np.dot(np.diag(sigmaB[:self.precent]), VB[:self.precent, :]))
    
   
        output = cv2.merge([np_a_approx_R, np_a_approx_G, np_a_approx_B])
        return output
######################################################################################################################
#######                                                                      #######
######################################################################################################################
def GetSecretAcc(secret_true, secret_pred):
    if 'cuda' in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    str_acc = 1.0 - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy() / correct_pred.size()[0]
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()
    return bit_acc, str_acc

######################################################################################################################
#######                                                                      #######
######################################################################################################################
class qrCode_reader():
    """ 
    """
    def __init__(self):
        
        self.readers_fun = lambda x: tf.map_fn(self.readerfn, x)
    def readerfn(self, image):
        # initialize the cv2 QRCode detector
        detector = cv2.QRCodeDetector()
        #print(image)
        data, bbox, _ = detector.detectAndDecode(np.array(image))
        # check if there is a QRCode in the image
        if bbox is not None:
            # display the image with lines
            for i in range(len(bbox)):
                # draw all lines
            
                if data:
                    return 1, data
                else:
                    return 0, data
        else:
            return 0, data
    def reads(self, images_batch):
        if True:
            acc, data = self.readerfn((images_batch[0]*255).numpy().astype("uint8"))
            #acc = self.readers_fun(acc)
        if False:
            acc= 0
            data=""
        return acc, data
    

######################################################################################################################
#######                                                                      #######
######################################################################################################################
    

def get_secret_acc(secret_true, secret_pred):
    #if 'cuda' in str(secret_pred.device):
    #    secret_pred = secret_pred.cpu()
    #    secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    str_acc = 1.0 - torch.sum((correct_pred - secret_pred.size()[1]) != 0)/ correct_pred.size()[0]
    bit_acc = torch.sum(correct_pred) / secret_pred.numel()
    return bit_acc, str_acc