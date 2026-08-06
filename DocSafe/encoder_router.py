
"""
@--01.04.2026--@
Author: github/farhadsh1992
INFO:
LAST_UPDATE:
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from FarhadCV.Tools import tcolors, bcolors
import bchlib
from PIL import Image, ImageOps
from .Tools import Convert_String_To_Binery_Message

class  encoder():
    def __init__(self, 
                model:str="M1", # M1, M2, M3 
                path_model:str="./pre_trained_models/", 
                secret_size:int=100,
                save_dir:str=None,
                batch_size:int = 1,

                BCH_BITS       = 7, # 13, 7, 21, 44
                BCH_POLYNOMIAL = 137, #487#137#8219, 20023 
                number_zeros   = 4,
                devices:str="-1",
                ):
        super(encoder, self).__init__()

        ## parameters
        self.model = model
        self.path_model = path_model
        self.secret_size = secret_size
        self.original_image = None
        self.binery_messages = None
        self.gpu_devices    = devices
        self.save_dir = save_dir
        
        self.batch_size = batch_size
        self.image_size = 256
        self.croper_size  = 64
        self.image_shape  = (256, 256)
        self.image_shape_de  = (256, 256)
        self.message_shape   = (16, 16)
        self.message_size    = 16
        self.mess_reshape    = (10, 10)
        self.secret_size   = 100
        self.pad_size      = 3
        self.length_str_message   = 12
        self.lenght_digit_message = 12
        self.image_size_de        = 256

        self.BCH_BITS       = BCH_BITS # 13, 7, 21, 44
        self.BCH_POLYNOMIAL = BCH_POLYNOMIAL #487#137#8219, 20023 
        self.number_zeros   =  number_zeros
        self.sevensize      =  1
        self.bits           = secret_size

        ## setup model parameters based on the selected model
        if self.model == "M1":
            ## StampOneDetr2_SIRIN5_CompleteFFT_X5010"
            self.name_project = "StampOneDetr2_SIRIN5_CompleteFFT_X5010"
            self.name_model   = "DocSafe_M1"
            self.steps = str(364000)

            self.path_model = path_model + "M1_models/"
            self.path_detr_en_weights   = (self.path_model + f"Steps" + str(self.steps) + "_" + "DetrNetEn"+"_"+ self.name_project + '.pt')
            self.path_detr_de_weights   = (self.path_model + f"Steps" + str(self.steps) + "_" + "DetrNetDe"+"_"+ self.name_project + '.pt')
            
            self.path_encoder_weights = (self.path_model + "Steps" + str(self.steps) + "_" + "EncoderNet"+"_"+  self.name_project + ".pt") 
            self.path_decoder_weights = (self.path_model + "Steps" + str(self.steps) + "_" + "DecoderNet"+"_"+ self.name_project  + ".pt")
     
        elif self.model == "M2":
            ## StampOneGuard10_HUM10_CFFT_XZ1020"
            self.name_project = "StampOneGuard10_HUM10_CFFT_XZ1020"
            self.name_model   = "DocSafe_M2"
            self.steps = str(230000)

            self.path_model = path_model + "M2_models/"
            self.path_detr_weights1   = (self.path_model + f"Steps" + str(self.steps) + "_" + "DetrNet"+"_"+ self.name_project + '.pt')
            self.path_encoder_weights = (self.path_model + "Steps" + str(self.steps) + "_" + "EncoderNet"+"_"+  self.name_project + ".pt") 
            self.path_decoder_weights = (self.path_model + "Steps" + str(self.steps) + "_" + "DecoderNet"+"_"+ self.name_project  + ".pt")


        elif self.model == "M3":
            ## StampOneGuard10_HUM10_CFFT_XZ1021"
            self.name_project = "StampOneGuard10_HUM10_CFFT_XZ1021"
            self.name_model   = "DocSafe_M3"
            self.steps = str(340000)

            self.path_model = path_model + "M3_models/"
            self.path_detr_weights1   = (self.path_model + f"Steps" + str(self.steps) + "_" + "DetrNet"+"_"+ self.name_project + '.pt')
            self.path_encoder_weights = (self.path_model + "Steps" + str(self.steps) + "_" + "EncoderNet"+"_"+  self.name_project + ".pt") 
            self.path_decoder_weights = (self.path_model + "Steps" + str(self.steps) + "_" + "DecoderNet"+"_"+ self.name_project  + ".pt")

        print(tcolors.GREEN+tcolors.BOLD)
        print("PROJECT: ", self.name_project)
        print("MODEL: ", self.name_model, "| STEPS:", self.steps)
        print(tcolors.ENDC)

    def read_image(self, path:list):
        ## read image from path
        self.original_image = []
        for p in path:
            img = Image.open(p).convert('RGB')
            self.original_image.append(img)
        return self.original_image
    def preprocess_images(self, images):
        ## preprocess images
        image_batch, self.image_batch2 = [], []
        self.width_list = []
        self.height_list = []

        for img in images:
            self.width, self.height = img.size
            self.width_list.append(self.width)
            self.height_list.append(self.height)


            img1 = self.transform_cover(img)
            image_batch.append(img1)

            img2 = self.transform_cover2(img)
            self.image_batch2.append(img2)

        image_batch = torch.stack(image_batch, dim=0)
        return image_batch
    def preprocess_messages(self, messages):    
        ## preprocess messages
        binery_messages = []
        for msg in messages:
            bin_msg = self.read_message(msg)
            binery_messages.append(bin_msg)
        binery_messages = torch.stack(binery_messages, dim=0)
        return binery_messages

    def load_transformers(self):

        ########################################################################
        ####                             ####
        ########################################################################
        self.transform_cover = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        self.transform_cover2 = transforms.Compose([
                    # transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
        self.transform_message = transforms.Compose([
                transforms.Resize((16, 16)),
                transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        # self.encoded_transform = transforms.Compose([
        #             transforms.Resize((256, 256)),
        #             transforms.ToTensor(),
        #             # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #             ])
        ########################################################################
        ####                             ####
        ########################################################################
        self.bch_router = bchlib.BCH(self.BCH_BITS, self.BCH_POLYNOMIAL)
        self.read_message = read_message
        ########################################################################
    def load_network(self, device=None):
        if self.model == "M1":
            self.Load_M1_networks(device=device)
        elif self.model == "M2":
            self.Load_M2_networks(device=device)
        elif self.model == "M3":
            self.Load_M3_networks(device=device)
        ## Load pre-trained transformers
        self.load_transformers()

        self.message_router = Convert_String_To_Binery_Message(
                BCH_BITS = self.BCH_BITS,
                BCH_POLYNOMIAL = self.BCH_POLYNOMIAL,
                number_zeros = self.number_zeros,
                sevensize = self.sevensize,
                mess_reshape = self.mess_reshape,
                message_shape = self.message_shape, 
                pad_size = self.pad_size, 
                message_transform=self.transform_message)
        
    def Load_M1_networks(self, device=None):
        self.device = device
        
        
        
        self.detr_en = None
        from .networks.networks_M1.AttentionVNet_encoder import AttentionVnetEncoder
        self.EncoderNet = AttentionVnetEncoder(
                            detr_load     = self.detr_en, 
                            batch_size    = self.batch_size, 
                            image_shape   = self.image_shape,
                            message_shape = self.message_shape,
                            croper_size   = self.croper_size,
                            device        = device).to(device)

        ######################################################################################
        # self.detr_de = None
        # from .networks.networks_M1.AttentionVNet_decoder import AttentionVnetDecoder
        # self.DecoderNet = AttentionVnetDecoder(
        #                             detr_load   = self.detr_de, 
        #                             batch_size  = self.batch_size,  
        #                             image_shape = self.image_shape_de, 
        #                             croper_size = self.croper_size,
        #                             device      = device).to(device)
        ######################################################################################
        ## Load pre-trained encoder
        en_path = torch.load(self.path_encoder_weights, map_location=torch.device('cpu'))
        self.EncoderNet.load_state_dict(en_path)
        ## Load pre-trained decoder
        # de_apth = torch.load(self.path_decoder_weights, map_location=torch.device('cpu'))
        # self.DecoderNet.load_state_dict(de_apth)
        ######################################################################################
        

    def Load_M2_networks(self, device=None):



        self.device = device
        
        self.detr_en = None
        from .networks.networks_M2.AttentionVNet_encoder import AttentionVnetEncoder
        self.EncoderNet = AttentionVnetEncoder(
                            detr_load     = self.detr_en, 
                            batch_size    = self.batch_size, 
                            image_shape   = self.image_shape,
                            message_shape = self.message_shape,
                            croper_size   = self.croper_size,
                            device        = device)#.to(device)


        # self.detr_de = None
        # from  .networks.networks_M2.AttentionVNet_decoder import AttentionVnetDecoder
        # self.DecoderNet = AttentionVnetDecoder(
        #                             detr_load   = self.detr_de, 
        #                             batch_size  = self.batch_size,  
        #                             image_shape = self.image_shape_de, 
        #                             croper_size = self.croper_size,
        #                             device      = device)#.to(device)


        ######################################################################################
        ## Load pre-trained encoder
        en_path = torch.load(self.path_encoder_weights, map_location=torch.device('cpu'))
        self.EncoderNet.load_state_dict(en_path)
        ## Load pre-trained decoder
        # de_apth = torch.load(self.path_decoder_weights, map_location=torch.device('cpu'))
        # self.DecoderNet.load_state_dict(de_apth)

    def Load_M3_networks(self, device=None):
        self.device = device
        
       
       
       

        self.detr_en = None
        from .networks.networks_M3.AttentionVNet_encoder import AttentionVnetEncoder
        self.EncoderNet = AttentionVnetEncoder(
                            detr_load     = self.detr_en, 
                            batch_size    = self.batch_size, 
                            image_shape   = self.image_shape,
                            message_shape = self.message_shape,
                            croper_size   = self.croper_size,
                            device        = device).to(device)

        # self.detr_de = None
        # from ..networks.networks_M3.AttentionVNet_decoder import AttentionVnetDecoder
        # self.DecoderNet = AttentionVnetDecoder(
        #                             detr_load   = self.detr_de, 
        #                             batch_size  = self.batch_size,  
        #                             image_shape = self.image_shape_de, 
        #                             croper_size = self.croper_size,
        #                             device      = device).to(device)


        ######################################################################################
        ## Load pre-trained encoder
        en_path = torch.load(self.path_encoder_weights, map_location=torch.device('cpu'))
        self.EncoderNet.load_state_dict(en_path)
        ## Load pre-trained decoder
        # de_apth = torch.load(self.path_decoder_weights, map_location=torch.device('cpu'))
        # self.DecoderNet.load_state_dict(de_apth)

    def __call__(self, 
                original_images, messages, mask=None,
                coefficient=1.0,
                ):



        ## preprocess messages
        batch_message_in, batch_message_gray =  self.message_router(messages)
        # batch_message_in = self.preprocess_messages(messages)
        self.binery_messages = batch_message_in

        ## preprocess messages
        if mask is None:
            depth_in = torch.ones(original_images.shape)
        else:
            depth_in = self.preprocess_image(mask)

        images_in = F.interpolate(original_images, 
                                        size=(256, 256), 
                                        mode="nearest")
            
        #############################################################
        ## encoding images ... 
        (resduial, embedding, ximage) = self.EncoderNet(
                                    images  = images_in.to(self.device), 
                                    secrets = batch_message_in.to(self.device),
                                    mask    = depth_in.to(self.device))
        self.embedding = embedding
        self.ximage = ximage


        self.resduial = F.interpolate(resduial, 
                                        size=(self.height,self.width), 
                                        mode="nearest")#.clamp_(0,1)
        
        ## Contact resize_resduial with original image in original size
        encoded_image_float = np.clip(coefficient * self.resduial[0].cpu().detach().numpy() 
                        + (np.array(self.image_batch2[0].cpu().detach()).astype("float32"))#.transpose(2,0,1))
                        , 0,1)
        self.encoded_image = (encoded_image_float*255.0).astype("uint8")#.transpose(1, 2, 0)
        # output: encoded_image, self.resduial, encoded_image_float, self.binery_messages


        return self.encoded_image
    def save_encoded_image(self, save_path:str):
        if self.encoded_image is not None:
            cv2.imwrite(save_path, self.encoded_image)
            print(tcolors.GREEN, "Encoded image saved at: ", save_path, tcolors.ENDC)
        else:
            print(tcolors.RED, "No encoded image to save.", tcolors.ENDC)


##################################################################################
####                                     ####
##################################################################################  
def read_message(Message, BCH_BITS, BCH_POLYNOMIAL, secret_size,mess_reshape, number_zeros, pad=0):
    ########################################
    len_images     = Message.shape[0]
    BCH_BITS       = BCH_BITS
    BCH_POLYNOMIAL = BCH_POLYNOMIAL
    bits           = secret_size
    shape_message  = mess_reshape
    number_zeros   = number_zeros

    bch_model = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL)
    # bch_model = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    ########################################
    # Message = (Message+0.5)*255.0
    if pad != 0:
        Message = Message[:, :, pad:-1*pad, pad:-1*pad]

    Message = F.interpolate(Message, 
                                size=(shape_message[0], shape_message[1]), 
                                mode="nearest")#.clamp_(0,1)
    Message = RGBToGrayLayer()(Message)
    Message = (Message*255.0).cpu().detach().numpy().astype("uint8").transpose(0, 2, 3, 1)
    Message = np.where((Message > 127.5), 255, Message)
    Message = np.where((Message < 127.5), 0, Message)
    Message = (Message.astype("float32")/255.0).astype("uint8")


        
    Message2 = np.reshape(Message, (len_images, 1*shape_message[0]*shape_message[1])).astype("uint8")
    # print(Message2.shape)

    list_decoded_msg = []
    for i in range(len_images):
        try:
            decoded_msg = BCH_Reader(
                                    bch_model      = bch_model, 
                                    secret         = Message2[i], 
                                    BCH_BITS       = BCH_BITS  , 
                                    BCH_POLYNOMIAL = BCH_POLYNOMIAL, 
                                    bits           = bits,  
                                    number_zeros   = number_zeros)
            # print(tcolors.BLUE, f"\n ({i}) message:",decoded_msg, tcolors.ENDC)
        except Exception as error:
            decoded_msg = "-NONE-"
            print(tcolors.RED, f"\n ({i}) error:",error, tcolors.ENDC)
        list_decoded_msg.append(decoded_msg)
    return Message , list_decoded_msg


##################################################################################
####                                     ####
##################################################################################
def BCH_Reader(bch_model, secret, BCH_BITS, BCH_POLYNOMIAL, bits = 96, number_zeros=4):
    # import bchlib
   
        
  
    #print(secret)
    # bch = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL)
    bch = bch_model ##bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)


    bitss = bits-number_zeros
    packet_binary = "".join([str(int(bit)) for bit in secret[:bitss]])
    packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
    packet = bytearray(packet)
    
    
    data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
    # bitflips = bch.decode_inplace(data, ecc)
    bitflips = bch.decode(data, ecc)
    try:
        decoded_msg = data.decode("utf-8")
        # print( tcolors.GREEN, "\n message:",decoded_msg, tcolors.ENDC)
            
    except Exception as error:
        # print( tcolors.RED, "error", error, tcolors.ENDC)
        decoded_msg = "none"
        
    return decoded_msg
##################################################################################
####                                     ####
##################################################################################
def BCH_Generator(bch, secret, 
                  BCH_POLYNOMIAL = 137, 
                  BCH_BITS = 7, 
                  number_zeros = 4, 
                  sevensize = 1):

    #print(tcolors.RED, secret, tcolors.ENDC)
    # bch    = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL)
    data   = bytearray(secret + ' '*(sevensize-len(secret)), 'utf-8')
    ecc    = bch.encode(data)
    packet = data + ecc
    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0 for i in range(number_zeros)])
    return secret