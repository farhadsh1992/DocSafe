"""
@--01.04.2026--@
Author: github/farhadsh1992
INFO:
LAST_UPDATE:
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import bchlib
from PIL import Image, ImageOps
from FarhadCV.Tools import tcolors, bcolors



######################################################################
#######                                                  #######
#######################################################################
class Convert_String_To_Binery_Message(nn.Module):
    def __init__(self, 
            BCH_BITS,
            BCH_POLYNOMIAL,
            number_zeros,
            sevensize,
            mess_reshape,
            message_shape, 
            pad_size, 
            message_transform):
        super(Convert_String_To_Binery_Message, self).__init__()
        self.BCH_BITS       = BCH_BITS
        self.BCH_POLYNOMIAL = BCH_POLYNOMIAL
        self.number_zeros   = number_zeros
        self.sevensize      = sevensize
        if mess_reshape != None:
            self.reshape_size_1 = mess_reshape[0]
            self.reshape_size_2 = mess_reshape[1]
            self.image_size_qr1 = message_shape[0]
            self.image_size_qr2 = message_shape[1]
        else:
            self.reshape_size_1 = None
            self.reshape_size_2 = None
            self.image_size_qr1 = None
            self.image_size_qr2 = None

        self.bch = bchlib.BCH(self.BCH_BITS, self.BCH_POLYNOMIAL)

        self.message_transform = message_transform

        self.pad_size_U = pad_size
        self.pad_size_D = pad_size
        self.pad_size_L = pad_size
        self.pad_size_R = pad_size
    def forward(self, secrets:list[str]):
        
        list_binery2D = []
        list_binery2D_gray = []
   
        print(tcolors.GREEN, "BCH_Generator: ", secrets, tcolors.ENDC)
        secret = BCH_Generator(
                            bch = self.bch, 
                            secret = secrets[0], 
                            BCH_POLYNOMIAL = self.BCH_POLYNOMIAL, 
                            BCH_BITS = self.BCH_BITS, 
                            number_zeros = self.number_zeros, 
                            sevensize = self.sevensize)

        ########################################################################
        # print(tcolors.GREEN, "BCH_Generator: ", secret, tcolors.ENDC)
        if self.reshape_size_1 != None and self.reshape_size_2 != None:
            MessageView = np.reshape(secret, (self.reshape_size_1, self.reshape_size_2, 1))
            # print("MessageView: ", MessageView.shape)
        else:
            MessageView = secret

        ########################################################################
        a_vector_message2D = np.uint8(MessageView)*255
        ########################################################################
        if self.reshape_size_1 != None and self.reshape_size_2 != None:
            # print("a_vector_message2D: ",a_vector_message2D.shape)
            # a_vector_message2D = (a_vector_message2D/255.0).astype('float32')
            a_vector_message2D = cv2.cvtColor(a_vector_message2D, cv2.COLOR_GRAY2BGR)
            a_vector_message2D_gray = cv2.cvtColor(a_vector_message2D, cv2.COLOR_BGR2GRAY)

            a_vector_message2D = cv2.copyMakeBorder(a_vector_message2D, 
                                                    top = self.pad_size_U, 
                                                    bottom = self.pad_size_D, 
                                                    left = self.pad_size_L, 
                                                    right = self.pad_size_R, 
                                                    borderType = cv2.BORDER_CONSTANT, 
                                                    value = (255,255,255))
            a_vector_message2D_gray = cv2.copyMakeBorder(a_vector_message2D_gray, 
                                                    top = self.pad_size_U, 
                                                    bottom = self.pad_size_D, 
                                                    left = self.pad_size_L, 
                                                    right = self.pad_size_R, 
                                                    borderType = cv2.BORDER_CONSTANT, 
                                                    value = (255,255,255))
            a_vector_message2D = cv2.resize(a_vector_message2D, 
                                            (self.image_size_qr1, self.image_size_qr2), 
                                            interpolation=cv2.INTER_NEAREST)
            a_vector_message2D_gray = cv2.resize(a_vector_message2D_gray, 
                                            (self.image_size_qr1, self.image_size_qr2), 
                                            interpolation=cv2.INTER_NEAREST)
        else:
            a_vector_message2D = a_vector_message2D
            a_vector_message2D_gray = a_vector_message2D
        ########################################################################
        a_vector_message2D =  Image.fromarray(np.uint8(a_vector_message2D))
        a_vector_message2D_gray =  Image.fromarray(np.uint8(a_vector_message2D_gray))
        
        message_in_en = self.message_transform(a_vector_message2D)
        message_in_en_gray = self.message_transform(a_vector_message2D_gray)
        

        list_binery2D.append(message_in_en)
        list_binery2D_gray.append(message_in_en_gray)

        list_binery2D = torch.stack(list_binery2D)
        list_binery2D_gray = torch.stack(list_binery2D_gray)

        # print(tcolors.GREEN, "list_binery2D: ", list_binery2D.shape, tcolors.ENDC)
        # print(tcolors.GREEN, "list_binery2D_gray: ", list_binery2D_gray.shape, tcolors.ENDC)
        return list_binery2D, list_binery2D_gray


#######################################################################
#######                                                  #######
#######################################################################
def BCH_Generator(bch, secret, BCH_POLYNOMIAL = 137, BCH_BITS = 7, number_zeros = 4, sevensize = 1):

    #print(tcolors.RED, secret, tcolors.ENDC)
    # bch    = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL)
    data   = bytearray(secret + ' '*(sevensize-len(secret)), 'utf-8')
    ecc    = bch.encode(data)
    packet = data + ecc
    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0 for i in range(number_zeros)])
    return secret



class RGBToGrayLayer(nn.Module):
    def __init__(self):
        super(RGBToGrayLayer, self).__init__()

    def forward(self, x):
        # Ensure the input has 3 channels (RGB)
        if x.shape[1] != 3:
            print(tcolors.RED, "Input Shape: ", x.shape, tcolors.ENDC)
            raise ValueError("Input must have 3 channels (RGB)")

        # Apply the standard luminosity method for RGB to Grayscale conversion
        # Weights are based on the human perception of colors: 0.2989 * R + 0.5870 * G + 0.1140 * B
        r, g, b = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray