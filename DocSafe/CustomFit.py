



"""
@--04.02.2025--@
Author: github/farhadsh1992
INFO:
    -ref: 
        
    
LAST_UPDATE:
"""





################################################
### pyTorch
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import keras
import numpy as np
################################################
from Network_Libs.ArtCoder.utils import gram_matrix
from Network_Libs.ArtCoder.utils import get_action_matrix
from Network_Libs.ArtCoder.utils import tensor_to_PIL
from FarhadCV.Tools import tcolors, bcolors, estimator
################################################
from .loss_functions_lib.tools import get_secret_acc
from Tools_GAN.linear_interpolation import blend_images
from Tools_nvidia_torch.torch_utils import check_free_space


###############################################################
# Full model.
class Trainer(nn.Module):
    """
    -----------------------------------------------------
    INPUTS:
        
    -----------------------------------------------------
    OUTPUTS:
        
    -----------------------------------------------------
    INFO:
        
    -----------------------------------------------------
    """
    def __init__(self, 
                model:str         = "M1",  ## M1, M2, M3
                path_model:str    = "./pre_trained_models/",
                args        = None,
                args_noise = None,
                devices = "gpu:0",
                 
            ):
        super(Trainer, self).__init__()
        

        ##############################################################################################
        #######     PARAMETRES                                                    # ######
        #############################################################################################
        self.args          =  args
        self.model           =  model
        self.args_noise    = args_noise
        self.batch_size    = args.batch_size
        self.image_size    = args.image_size
        self.message_shape = args.message_shape
        self.devices       = [devices]
        self.gpu_id        = devices
        # self.gpu_id        = gpu_id

        self.apply_detr      = args.apply_detr
        self.image_size      = args.image_size #self.model_size * self.model_num 
        self.image_shape     = ( args.image_size,  args.image_size)
        self.image_shape_de  = (args.image_size_de, args.image_size_de)
        self.borders         = args.borders

        
        self.shape_message = args.message_shape
        self.croper_size   = args.croper_size
        ##############################################################################################
        #######             Load DETR Models                                           # ######
        #############################################################################################
        if self.apply_detr:
            from networks.DETR_NET.detr_net import detr_resnet50
            self.detr_en = detr_resnet50(pretrained  = False, 
                                        num_classes = 1, 
                                        num_queries = 1,
                                        return_postprocessor = False, 
                                        device=devices[0]).to(self.devices[0])
            self.detr_de = detr_resnet50(pretrained  = False, 
                                        num_classes = 1, 
                                        num_queries = 1,
                                        return_postprocessor = False, 
                                        device=devices[0]).to(self.devices[0])
        else:
            self.detr_en = None
            self.detr_de = None


        ##############################################################################################
        #######                                                         # ######
        #############################################################################################
        if self.model == "M1":
            self.Load_M1_networks(device=devices[0])
        elif self.model == "M2":
            self.Load_M2_networks(device=devices[0])
        elif self.model == "M3":
            self.Load_M3_networks(device=devices[0])

        ##############################################################################################
        #######                                                         # ######
        #############################################################################################
        from .networks.net_StampOne.Stega_Discrimnator import StegaDiscriminator
        self.stega_disc    = StegaDiscriminator(device=self.devices[0])

     
        from .networks.net_StampOne.Specteral_Discrimnator_v03 import SpectralDiscriminatorRouter2
        self.spectral_disc = SpectralDiscriminatorRouter2(
                                                batch_size = self.batch_size,
                                                height     = self.shape_message[0],
                                                device     = self.devices[0])
        
        ##############################################################################################
        #######                                                         # ######
        #############################################################################################
        if self.args.load_pre_model:
            self.load_models()
   
        ##############################################################################################
        #######                                                         # ######
        #############################################################################################
        self.term_track  = {}
        self.term_tracki = {}
    def Load_M1_networks(self, device=None):
        self.device = device
        
        
        
        # self.detr_en = None
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
        from .networks.networks_M1.AttentionVNet_decoder import AttentionVnetDecoder
        self.DecoderNet = AttentionVnetDecoder(
                                    detr_load   = self.detr_de, 
                                    batch_size  = self.batch_size,  
                                    image_shape = self.image_shape_de, 
                                    croper_size = self.croper_size,
                                    device      = device).to(device)
        ######################################################################################
        ## Load pre-trained encoder
        # en_path = torch.load(self.path_encoder_weights, map_location=torch.device('cpu'))
        # self.EncoderNet.load_state_dict(en_path)
        ## Load pre-trained decoder
        # de_apth = torch.load(self.path_decoder_weights, map_location=torch.device('cpu'))
        # self.DecoderNet.load_state_dict(de_apth)
        ######################################################################################
        

    def Load_M2_networks(self,  device=None):
        self.device = device



       
        
        # self.detr_en = None
        from .networks.networks_M2.AttentionVNet_encoder import AttentionVnetEncoder
        self.EncoderNet = AttentionVnetEncoder(
                            detr_load     = self.detr_en, 
                            batch_size    = self.batch_size, 
                            image_shape   = self.image_shape,
                            message_shape = self.message_shape,
                            croper_size   = self.croper_size,
                            device        = device)#.to(device)


        # self.detr_de = None
        from  .networks.networks_M2.AttentionVNet_decoder import AttentionVnetDecoder
        self.DecoderNet = AttentionVnetDecoder(
                                    detr_load   = self.detr_de, 
                                    batch_size  = self.batch_size,  
                                    image_shape = self.image_shape_de, 
                                    croper_size = self.croper_size,
                                    device      = device)#.to(device)


        ######################################################################################
        ## Load pre-trained encoder
        # en_path = torch.load(self.path_encoder_weights, map_location=torch.device('cpu'))
        # self.EncoderNet.load_state_dict(en_path)
        ## Load pre-trained decoder
        # de_apth = torch.load(self.path_decoder_weights, map_location=torch.device('cpu'))
        # self.DecoderNet.load_state_dict(de_apth)

    def Load_M3_networks(self, device=None):
        self.device = device
        
       
       
       

        # self.detr_en = None
        from .networks.networks_M3.AttentionVNet_encoder import AttentionVnetEncoder
        self.EncoderNet = AttentionVnetEncoder(
                            detr_load     = self.detr_en, 
                            batch_size    = self.batch_size, 
                            image_shape   = self.image_shape,
                            message_shape = self.message_shape,
                            croper_size   = self.croper_size,
                            device        = device).to(device)

        # self.detr_de = None
        from ..networks.networks_M3.AttentionVNet_decoder import AttentionVnetDecoder
        self.DecoderNet = AttentionVnetDecoder(
                                    detr_load   = self.detr_de, 
                                    batch_size  = self.batch_size,  
                                    image_shape = self.image_shape_de, 
                                    croper_size = self.croper_size,
                                    device      = device).to(device)


        ######################################################################################
        ## Load pre-trained encoder
        # en_path = torch.load(self.path_encoder_weights, map_location=torch.device('cpu'))
        # self.EncoderNet.load_state_dict(en_path)
        ## Load pre-trained decoder
        # de_apth = torch.load(self.path_decoder_weights, map_location=torch.device('cpu'))
        # self.DecoderNet.load_state_dict(de_apth)
    def load_loss_functions(self,
        ## encoder loss
        apply_lpips = True,
        apply_riemann = True,
        apply_color = True,
        apply_yuv = True,
        apply_mse = True,
        ## decoder loss
        apply_bce = True,
        apply_affine = False,
        ):
         
       
        
        ##############################################################################################
        #######      EncoderLoss                                                   # ######
        #######                                                                     #######
        #############################################################################################
        # from Network_Libs.ArtCoder.vgg_net import Vgg16
        # self.vgg_feature_extraction = Vgg16(requires_grad=False).to(self.devices[0])

        # from Network_Libs.ArtCoder.SS_layer import SSlayer
        # self.ss_layer = SSlayer(model_size=self.model_size, requires_grad=False).to(self.devices[0])

        #########################################################################
        ##  LOSS LPIPS        
        ## learned perceptual metric model
        from .networks.loss_functions_lib.lpips.lpips import LPIPS
        self.lpips_en_loss = LPIPS(input_size=256, net_type='alex', device=self.devices[0])#.to(self.devices[0]).eval()
        # self.lpips_de_loss = LPIPS(net_type='alex', device=self.devices[0]).to(self.devices[0]).eval()
        self.lpips_croper_loss = LPIPS(input_size=64, net_type='alex', device=self.devices[0])

        #########################################################################
        #####                #####
        #########################################################################
        # ### [2] Riemann Loss 
        if self.args.weight_riem_rec is not 0:
            ####################
            # from loss_functions_lib.Riemann_Loss_Torch_v1 import Riemann_Loss
            # self.Rieman_Loss_Router_en = Riemann_Loss(batch_size=self.batch_size, 
            #                                           image_size=self.image_size)
            # self.Rieman_Loss_Router_de = Riemann_Loss(batch_size=self.batch_size, 
            #                                           image_size=self.message_shape[0])
            ####################
            from .networks.loss_functions_lib.Riemann_Loss_Patches_Torch_v2 import Riemann_Patcher_Loss
            # from loss_functions_lib.Riemann_Loss_Patches_Torch_v3 import Riemann_Patcher_Loss3
            self.Rieman_Loss_Router_en = Riemann_Patcher_Loss(
                                            batch_size  = self.batch_size, 
                                            image_size  = self.image_size,
                                            patch_size  = 4,
                                            in_channels = 3,
                                            device      = self.devices[0]
                                                      ).to(self.devices[0])
            
            
            self.Rieman_Loss_Router_de = Riemann_Patcher_Loss(
                                            batch_size  = self.batch_size, 
                                            image_size  = self.message_shape[0],
                                            patch_size  = 1,
                                            in_channels = 1,
                                            device      = self.devices[0]
                                                      ).to(self.devices[0])
            ####################
            # from loss_functions_lib.query_selected_attention_wirth_Riemann_loss import QSModel
            # self.Rieman_Loss_Router_en = QSModel(
            #             batch_size = self.batch_size, 
            #             image_size = self.croper_size,
            #             net      = "net", 
            #             n_layers = 5,     
            #             kind_net = "vgg",
            #             device   = self.devices[0]).to(self.devices[0])
            # self.Rieman_Loss_Router_de = QSModel(
            #             batch_size = self.batch_size, 
            #             image_size = self.message_shape[0],
            #             net      = "net", 
            #             n_layers = 5,     
            #             kind_net = "vgg",
            #             device   = self.devices[0]).to(self.devices[0])
            ####################
        ##############################################
        ##    Color Loss for Encoder    ##
        ##############################################
        from .networks.loss_functions_lib.histogram_color_loss import RGBHistogram 
        # if self.args.weight_color_EN >0:
        self.colorRerouter = RGBHistogram(  h      = self.args.histogram_size, 
                                            insz   = self.args.max_input_size_hist, 
                                            method = self.args.method_his,
                                            sigma  = 0.02, 
                                            intensity_scale = True, 
                                            device = self.devices[0]
                                            ).to(self.devices[0])
        # self.colorRerouter_croper = RGBHistogram(  h      = self.args.histogram_size, 
        #                                     insz   = self.args.max_input_size_hist, 
        #                                     method = self.args.method_his,
        #                                     sigma  = 0.02, 
        #                                     intensity_scale = True, 
        #                                     device = self.devices[0]
        #                                     ).to(self.devices[0])
        #########################################################################
        from .networks.loss_functions_lib.yuv_loss import YUVLoss
        self.yuv_scales_pl = torch.tensor([self.args.y_scale, 
                                           self.args.u_scale, 
                                           self.args.v_scale], dtype=torch.float32)
        self.yuv_router = YUVLoss(input_size  = self.image_shape[0], 
                                yuv_scales_pl = self.yuv_scales_pl,
                                device        = self.devices[0]).to(self.devices[0])
        #########################################################################
        #####                 #####
        #########################################################################

        self.loss_mse = torch.nn.MSELoss(reduction='mean').to(self.gpu_id)
        #########################################################################
        #####                 #####
        #########################################################################
        # from loss_functions_lib.face_id_loss import FaceIDLoss
        # self.id_loss = FaceIDLoss(self.args).to(self.devices[1]).eval()

        # from loss_functions_lib.vgg_loss import VGGLoss 
        # self.vgg_loss_touter = VGGLoss(vgg_version="vgg19")

        # from loss_functions_lib.moco_loss import MocoLoss
        # self.moco_loss = MocoLoss()



        #########################################################################
        #####                 #####
        #########################################################################
        # self.loss_bce = torch.nn.BCELoss()
        # self.loss_bce = torch.nn.MSELoss()

        # self.m = torch.nn.Sigmoid()
        # self.loss_bce = torch.nn.BCEWithLogitsLoss()
        self.loss_bce = nn.BCELoss()#.to(self.devices[0])
        # Compute sigmoid cross-entropy loss (equivalent to TensorFlow)
        # self.loss_bce = F.binary_cross_entropy#_with_logits
        #########################################################################
        #####                 #####
        #########################################################################
        # from loss_functions_lib.query_selected_attention_loss import QSModel
        # self.qtAttn_router_de = QSModel(args = self.args, net = "net", 
        #                                 n_layers = 5,     kind_net = "vgg")
        #########################################################################
        #####       Affine_Loss          #####
        #########################################################################
        if apply_affine:
            from .networks.loss_functions_lib.Affine_Loss import AffineLoss
            self.affine_encoder_touter = AffineLoss(device=self.devices[0])
            self.affine_decoder_touter = AffineLoss(device=self.devices[0])

    def load_optimizers(self, 
        optimizer_type,  ## Adam or Ranger
        ):
        ##############################################
        # Initialize optimizer self.backre
        params_enc = list(self.EncoderNet .parameters())
        params_dec = list(self.DecoderNet .parameters())
        params_disc = list(self.stega_disc.parameters())
        params_Spec = list(self.spectral_disc.parameters())

        if self.args.apply_detr:
            params_detr_de  = list(self.detr_en.parameters())
            params_detr_en  = list(self.detr_de.parameters())

            params = params_enc + params_dec + params_Spec + params_detr_de + params_detr_en
        else:
            params = params_enc + params_dec + params_Spec

        ### Optimizers
        if optimizer_type == 'Adam':
            self.optimizer      = torch.optim.Adam(params, lr=self.args.learning_rate,  
                                                   betas=(0.50, 0.53))            
            self.disc_optimizer = torch.optim.Adam(params_disc, lr=self.args.learning_rate_disc,  
                                                   betas=(0.50, 0.53))
        elif optimizer_type == 'Ranger':
            from Tools_GAN.ranger import Ranger
            self.optimizer         = Ranger(params,      lr = self.args.learning_rate)
            self.disc_optimizer    = Ranger(params_disc, lr = self.args.learning_rate_disc)
           

    def perform_forward_on_batch(self, 
                                 cover_batch, 
                                 code_batch, 
                                 list_dapth,
                                 boxes,
                                 mask2, 
                                 gpu_id,
                                 epoch = 0):

        if self.args.apply_warper:
            self.warper_router.generate_random_matrix(epoch=epoch)
            cover_warper = self.warper_router.warper_inputs(cover_batch)
            dapth_warper = self.warper_router.warper_inputs(list_dapth)

        else:
            cover_warper = cover_batch
            self.borders = 'no_edge'


        (code_batch_en, code_batch_de) = code_batch
       
        ############################################################
        ####   Encoder              ####
        ############################################################
        self.optimizer.zero_grad()
        
        (residual_warped, embedding) = self.EncoderNet(images  = cover_warper, 
                                                mask    =  dapth_warper,
                                                secrets = code_batch_en)


        
        residual_warped = dapth_warper * residual_warped

        ############################################################
        ####   Apply Augmentor                    ####
        ############################################################
        if self.args.apply_warper:
            (encoded_image,  encoded_warped, residual, encoded_image2) = self.warper_router.warper_encoded_outputs(
                                                image           = cover_batch,
                                                input_warped    = cover_warper,
                                                residual_warped = residual_warped)
        else:
            encoded_image = encoded_warped = residual_warped + cover_batch
            residual      = residual_warped
        ############################################################
     
        encoded_image_in = encoded_image 

        if self.args.apply_noise:
            augmented_encoded_image = self.image_augmentor(encoded_image_in, 
                                                           steps = epoch)
        else:
            augmented_encoded_image = encoded_image
        ############################################################
        ####   Stega Discrimnator for ENC               ####
        ############################################################

        
        
        
        if self.borders == 'no_edge':
                D_output_real, _ = self.stega_disc(cover_batch)
                D_output_fake, D_heatmap = self.stega_disc(encoded_image)
        else:
            cover_warper1 = list_dapth * encoded_image
            encoded_warped1 = list_dapth * encoded_image
            D_output_real, _ = self.stega_disc(cover_warper)
            D_output_fake, D_heatmap = self.stega_disc(encoded_warped)
                

        D_loss = D_output_real #- D_output_fake
        xz = torch.from_numpy(D_output_fake.cpu().detach().numpy())
        disc_loss = (
                D_output_real - xz
                ).clamp_(-1e2, 1e5)
        G_loss = D_output_fake.clamp_(-1e3, 1e5)
        ############################################################
        ####   DECODER                ####
        ############################################################
    
        self.cof_mask = ramper_weight_minToMax(weight = 1, 
                                               ramper = self.args.mask_ramper, 
                                               epoch  = epoch)
        maskde = (1-self.cof_mask) * cover_warper + self.cof_mask * augmented_encoded_image
        # maskde = augmented_encoded_image

     
        (decoded_secret) = self.DecoderNet(augmented_encoded_image, maskde)

        ############################################################
        ####   Spectral Discrimnator  for REC              ####
        ############################################################
        spectral_loss = self.spectral_disc( decoded_secret, code_batch_de)


        ###############################################################################
        ##### Computing Total Loss, EncoderLoss and DecoderLoss #####
        ###############################################################################
        ## Load Ramper Cofficent
        self.ramper(epoch)
        ############################################################
        ####   ENCODER-Loss                ####
        ############################################################
        loss_lpips_en = self.lpips_en_loss(cover_batch, encoded_image2) 
        color_loss_en = self.colorRerouter(cover_batch, encoded_image2) 
        yuv_loss_en   = self.yuv_router(cover_batch, encoded_image2, 
                                        l2_edge_gain=self.args.l2_edge_gain)
        
       
        mse_loss_en = self.loss_mse(cover_batch, encoded_image2)
        loss_riem_en = self.Rieman_Loss_Router_en(cover_batch, encoded_image2)

   
        ############################################################
        ####   DECODER-Loss                ####src, tgt
        ############################################################
        
        bce_loss = self.loss_bce(decoded_secret, code_batch_de)
   

       
        loss_riem_de = self.Rieman_Loss_Router_de(code_batch_de, decoded_secret)
      
        ############################################################
        ####   Affine Loss             ####
        ############################################################
        # affine_loss_en = self.affine_encoder_touter(boxes, boxes_en)
        # affine_loss_de = self.affine_decoder_touter(boxes, boxes_de)
        
        ##########################################################
        #####    TOTAL LOSS #####
        ############################################################
        ## EncoderLoss
        encoder_loss = (   self.cof_lpips_enc * loss_lpips_en 
                         + self.cof_color_enc * color_loss_en 
                         + G_loss
                         + self.cof_yuv_enc   * yuv_loss_en
                         + self.cof_riem_enc * loss_riem_en
                         + self.cof_mse_enc * mse_loss_en
                        #  + self.cof_lpips_croper_enc * loss_lpips_cropper
                        #  + self.cof_color_enc * color_loss_croper
                         )
        ##########################################################
        # DecoderLoss
        decoder_loss = (
                          self.cof_bce              * bce_loss
                        + self.cof_spectral_rec     * spectral_loss
                        + self.cof_riem_rec         * loss_riem_de
                        )
        
        
        ##########################################################
        ## Total Loss
        
        if self.args.Just_train_decoder > epoch :
            # self.unfreeze_network(model = self.detr)
            total_loss = ( self.args.weight_rec * decoder_loss 
                        #   + affine_loss_en 
                        #   + affine_loss_de
                          )
        else :
            total_loss = ( self.args.weight_rec * decoder_loss 
                         + self.cof_enc         * encoder_loss
                        #  + affine_loss_en 
                        #  + affine_loss_de
                         )
            
        
      
        # total_loss = self.cof_rec * decoder_loss
        ###############################################################################
        #####                                             #####
        ###############################################################################
        
        ####################################
        
        ####################################
        total_loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(self.spectral_disc.parameters(), max_norm=1.0)
        self.optimizer.step()
        ####################################
        # Assuming D_loss is a tensor representing the discriminator loss
        self.disc_optimizer.zero_grad()  # Clear previous gradients

        
        disc_loss.backward(retain_graph=True)
        # Clip gradients for all trainable parameters of stega_disc
        # for param in self.stega_disc.parameters():
        #     if param.grad is not None:
        #         param.grad.data.clamp_(-0.25, 0.25)  # Gradient clipping
        torch.nn.utils.clip_grad_value_(self.stega_disc.parameters(), clip_value=0.25)
        # print(self.stega_disc.parameters())
        # Apply the gradients to update the model parameters
        self.disc_optimizer.step()
        
        ####################################
        bit_acc, str_acc = get_secret_acc(code_batch_de, decoded_secret)
        ###############################################################################
        #####                                             #####
        ###############################################################################
        self.term_tracki["cover_batch"]       = cover_batch
        self.term_tracki["code_batch_en"]     = code_batch_en
        self.term_tracki["cover_warper"]      = cover_warper
        self.term_tracki["recover_message"]   = decoded_secret
        self.term_tracki["embedding"]         = embedding[:,:3,:,:]
        self.term_tracki["encoded_image"]     = encoded_image2
        self.term_tracki["encoded_image1"]    = encoded_image

        self.term_tracki["resduial"]          = residual
        self.term_tracki["augmented_encoded"] = augmented_encoded_image
        self.term_tracki["warper_encoded"]    = encoded_warped
        self.term_tracki["dapth"]             = list_dapth
        self.term_tracki["code_batch_de"]     = code_batch_de
        # self.term_tracki["dapth_warper"]      = dapth_warper

        # self.term_tracki["croped_de"]       = croped_de
        # self.term_tracki["effine_en"]       = effine_en
        # self.term_tracki["cropped_mask"]    = cropped_mask
        ###################################################
        ## for encoder
        self.term_track["total_loss"] =  total_loss

        self.term_track["encoder_loss"] =  encoder_loss
        self.term_track["lpips_en"]     =  loss_lpips_en
        self.term_track["color_en"]     =  color_loss_en
        
        self.term_track["G_loss"]       =  G_loss
        self.term_track["yuv_en"]  =  yuv_loss_en
        self.term_track["mse_en"] = mse_loss_en
        self.term_track["covariance_en"]  =  loss_riem_en

        ###################################################
        ## For Decoder
        self.term_track["decoder_loss"]  = decoder_loss
        self.term_track["bce_loss"]      = bce_loss
        self.term_track["D_loss"]        =  D_loss
        self.term_track["covariance_de"]  =  loss_riem_de
        self.term_track["bit_acc"]       =  bit_acc
        self.term_track["str_acc"]       =  str_acc
        self.term_track["spectral_loss"] =  spectral_loss

        # self.term_track["affine_loss_en"] =  affine_loss_en
        # self.term_track["affine_loss_de"] =  affine_loss_de

        ###################################################
        self.term_track["total_loss"]    =  total_loss
       





        ##############################################
        ##                       ##
        ##############################################
        self.Live_Monitors.on_train_batch_end(epoch=epoch, 
                                              gpu_id = self.gpu_id, 
                                              logs=self.term_track, 
                                              logs2=self.term_tracki)
        
        
        ###################################################
        del(cover_batch)
        del(code_batch)
        # del(cover_warper)
        del(decoded_secret)
        del(cover_warper)
        del(embedding)
        del(encoded_image)
        del(residual)
        del(augmented_encoded_image)
        del(encoded_warped)
        del(bce_loss)
       



        return total_loss
    def _run_epoch(self, epoch):
    
        self.train_data.sampler.set_epoch(epoch)

        
        for iepoch, (iepoch2, cover_batch, (code_batch_en,code_batch_de), 
             dapth_batch, list_dapth, new_image) in enumerate(self.train_data):
       
            # cover_batch   = cover_batch[0].to(self.gpu_id)
            # code_batch_en = code_batch_en[0].to(self.gpu_id)
            # code_batch_de = code_batch_de[0].to(self.gpu_id)

            # dapth_batch = dapth_batch[0].to(self.gpu_id)
            # list_dapth = list_dapth[0].to(self.gpu_id)
            # new_image  = new_image[0].to(self.gpu_id)


           

            ##############################################

            self.perform_forward_on_batch(
                        cover_batch, 
                        (code_batch_en, code_batch_de), 
                        list_dapth,
                        # new_image = new_image,
                        boxes = None,
                        mask2 = None, 
                        gpu_id = self.gpu_id, 
                        epoch = epoch+iepoch)
            
            ##############################################
            ##                       ##
            ##############################################
            if epoch > 0:
                self.Live_Monitors.on_train_batch_end(
                                                epoch  = epoch, 
                                                gpu_id = self.gpu_id,
                                                logs   = self.term_track, 
                                                logs2  = self.term_tracki)
                ##############################################
                ##                       ##
                ##############################################
                if epoch%self.args.save_interval == 0:
                    self._save_checkpoint(epoch = epoch+iepoch)

            del(self.term_tracki)
            del(self.term_track)
            self.term_tracki = {}
            self.term_track = {}
            ##############################################
    def train_multi_gpus(self, max_epochs: int):
        command = (bcolors.BLUE+
            "TRAIN-"
            + tcolors.ENDC)
        
        self.Live_Monitors.on_train_begin()
        for epoch in range(self.args.initial_epoch, self.args.epochs):
            self._run_epoch(epoch)
           
    def train_with_one_gpu(self, max_epochs: int):
        command = (bcolors.BLUE+
            "TRAIN-"
            + tcolors.ENDC)
        
        self.Live_Monitors.on_train_begin()
        for epoch in range(self.args.initial_epoch, self.args.epochs):
            
            self.train_data.sampler.set_epoch(epoch)

        
            # for iepoch, (iepoch2, cover_batch, (code_batch_en,code_batch_de), 
            #  dapth_batch, list_dapth, new_image) in enumerate(self.train_data):
            # iepoch = int(iepoch[0])
            if True:
                (iepoch2, cover_batch, (code_batch_en,code_batch_de), 
                 dapth_batch, list_dapth, new_image) = next(iter(self.train_data))
                
                
                # print(tcolors.RED,"cover_batch: ",cover_batch.shape,tcolors.ENDC)
                # cover_batch   = cover_batch[0].to(self.gpu_id)
                # code_batch_en = code_batch_en[0].to(self.gpu_id)
                # code_batch_de = code_batch_de[0].to(self.gpu_id)

                # dapth_batch = dapth_batch[0].to(self.gpu_id)
                # list_dapth = list_dapth[0].to(self.gpu_id)
                # new_image  = new_image[0].to(self.gpu_id)


              

                ##############################################    
                self.perform_forward_on_batch(
                            cover_batch, 
                            (code_batch_en, code_batch_de), 
                            list_dapth,
                            # new_image = new_image,
                            boxes = None,
                            mask2 = None, 
                            gpu_id = self.gpu_id, 
                            epoch = epoch)
                
               
                if epoch > 0:
                    self.Live_Monitors.on_train_batch_end(
                                                    epoch  = epoch, 
                                                    gpu_id = self.gpu_id,
                                                    logs   = self.term_track, 
                                                    logs2  = self.term_tracki)
               
                ##############################################
                if epoch%self.args.save_interval == 0:
                    self._save_checkpoint(epoch = epoch)

                del(self.term_tracki)
                del(self.term_track)
                self.term_tracki = {}
                self.term_track = {}
                ##############################################
    def train(self, 
        rank: int, world_size: int,
        
        ):

        ## GPU Setting and multi-running
        if world_size > 1:
            ddp_setup(rank, world_size)
            device_one = torch.device(f'cuda:{rank}')
        elif world_size == 1: 
            rank = 0
            device_one = torch.device(f'cuda:{rank}')
        else:
            device_one = "cpu"

        ###################################################################################
        mkdirfile(args.log_dir)
        mkdirfile(args.save_dir + "/logs/")
        mkdirfile(args.save_dir + "/results/")
        mkdirfile(args.save_dir + "/results_final/")

        # mkdirfile(args.save_dir + "/Stega_Discrimnator/")
        # mkdirfile(args.save_dir + "/Spectral_Discrimnator/")
        # mkdirfile(args.save_dir + "/results/" + args.name_model + "/")
        mkdirfile(args.save_dir + "/results_final/" + args.name_model + "/")
        mkdirfile(args.save_dir + "/results_final/" + args.name_model + "/" + "Files/")
        mkdirfile(args.save_dir + "/results_final/" + args.name_model + "/" + "ckpts/")
        mkdirfile(args.save_dir + "/results_final/" + args.name_model + "/" + "Files/Networks_Libs/")
        mkdirfile(args.save_dir + "/results_final/" + args.name_model + "/" + "Files/NoiseSimulationPrime_Torch/")
        ###################################################################################
        print(tcolors.BLUE+bcolors.WHITE, "NAME-MODEL: ",   args.name_model,  tcolors.ENDC)
        print(tcolors.BLUE+bcolors.WHITE, "IMAGE-SIZE: ",   args.image_size,  tcolors.ENDC)
        print(tcolors.BLUE+bcolors.WHITE, "MESSAGE-SIZE-DE: ", args.message_shape,  tcolors.ENDC)
        print(tcolors.BLUE+bcolors.WHITE, "SECRET-BITS-DE: ", args.secret_size,  tcolors.ENDC)


        print(tcolors.BLUE+bcolors.WHITE, "BATCHS: ", args.batch_size, tcolors.ENDC)
        print(tcolors.BLUE+bcolors.WHITE, "EPOCHS: ", args.epochs, tcolors.ENDC)

        if self.args.distributed:
            self.train_multi_gpus(max_epochs)
        else:
            self.train_with_one_gpu(max_epochs)
    def load_data(self, 
                data_path="./data/", 
                val_path ="./data/", 
                mask_path="./data/masks/"):

            from .datasets.Loaddataset_v7 import Dataset_Router
            # if args.using_pad_max:
            # from datasets_lib.Loaddataset_v5 import Dataset_Router
            self.train_data, self.val_data = Dataset_Router(
                                args        = self.args, 
                                TRAIN_COVER = data_path,
                                TEST_COVER  = val_path,
                                rank        = self.args.rank, # which GPU this process is running on.
                                world_size  = self.args.world_size, # number of GPUs you want to use
                                device      = self.devices[0]
                                ).load()



    def upload_live_monitor(self, Live_Monitors):
        self.Live_Monitors=Live_Monitors
    def upload_augmentation(self, augmentor, warper_router):
        self.image_augmentor = augmentor
        self.warper_router   = warper_router

    def _save_checkpoint(self, epoch):

        
        try:
            assert check_free_space("./")["Free Space (GB)"] > 0.1
            torch.save(self.EncoderNet.state_dict(), 
                self.args.save_dir + 
                "results/"+ "Steps"+ str(epoch) + "_" + "EncoderNet" + "_" + self.args.name_model + ".pt")
            torch.save(self.DecoderNet.state_dict(), 
                self.args.save_dir + 
                "results/"+ "Steps"+ str(epoch) + "_" + "DecoderNet"+"_"+ self.args.name_model + ".pt")
            torch.save(self.stega_disc.state_dict(), 
                self.args.save_dir + 
                "results/"+ "Steps"+ str(epoch) + "_" + "StegaDisc"+"_"+ self.args.name_model  + ".pt")
            torch.save(self.spectral_disc.state_dict(), 
                self.args.save_dir + 
                "results/"+ "Steps"+ str(epoch) + "_" + "SpectralDisc"+"_"+self.args.name_model + ".pt")
            if self.args.apply_detr:
                torch.save(self.detr_en.state_dict(), 
                    self.args.save_dir + 
                    "results/"+ "Steps"+ str(epoch) + "_" + "DetrNetEn" + "_" + self.args.name_model + ".pt")
                torch.save(self.detr_de.state_dict(), 
                    self.args.save_dir + 
                    "results/"+ "Steps"+ str(epoch) + "_" + "DetrNetDe" + "_" + self.args.name_model + ".pt")
        
        except Exception as e:
            print(tcolors.RED,"The drive is full (90-CustomFit)!!",tcolors.ENDC)
            print("An error occurred:", e)
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'epoch': epoch,
        #     'loss': loss
        #     }, 'checkpoint.pth')

        # # Loading checkpoint
        # checkpoint = torch.load('checkpoint.pth')
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    def load_models(self):
        #############################################################
        ####                                                ####
        #############################################################
        try:
            assert check_free_space("./")["Free Space (GB)"] > 0.1
            print(tcolors.BLUE+bcolors.WHITE+
                "<Load Pre-Trained Models>", 
                tcolors.ENDC)
            ## Load pre-trained encoder
            en_path = torch.load(self.args.path_encoder_weights)
            self.EncoderNet.load_state_dict(en_path)
            ## Load pre-trained decoder
            de_apth = torch.load(self.args.path_decoder_weights)
            self.DecoderNet.load_state_dict(de_apth)
            ## Load pre-trained Stega-discrimnator
            stega_path = torch.load(self.args.path_stegadisc_weights)
            self.stega_disc.load_state_dict(stega_path)
            ## Load pre-trained Spectral-discrimnator
            spectral_path = torch.load(self.args.path_spectraldisc_weights)
            self.spectral_disc.load_state_dict(spectral_path)
        except Exception as e:
            print(tcolors.RED,"The drive is full (90-CustomFit)!!",tcolors.ENDC)
            print("An error occurred:", e)
        #############################################################
        ####                                                ####
        #############################################################
    # def _save_checkpoint(self, epoch):


    def ramper(self, epoch):
        #############################################################
        ####                                                ####
        #############################################################
        self.cof_rec = ramper_weight_minToMax(self.args.weight_rec, self.args.ramper_rec, epoch = epoch)
        self.cof_bce = ramper_weight_minToMax(self.args.weight_bce_rec, self.args.ramper_bce_rec, epoch = epoch)
        self.cof_lpips_rec = ramper_weight_minToMax(self.args.weight_lpips_rec, 
                                                    self.args.ramper_lpips_rec, 
                                                    epoch = epoch)
        self.cof_spectral_rec = ramper_weight_minToMax(self.args.weight_spectral_rec, 
                                                    self.args.ramper_spectral_rec, 
                                                    epoch = epoch)
        self.cof_riem_rec = ramper_weight_minToMax(self.args.weight_riem_rec, 
                                                    self.args.ramper_riem_rec, 
                                                    epoch = epoch)
        #############################################################
        ####                                                ####
        #############################################################
        self.cof_enc = ramper_weight_minToMax(self.args.weight_enc, self.args.ramper_enc, epoch = epoch)
        self.cof_lpips_enc = ramper_weight_minToMax(self.args.weight_lpips_enc, 
                                                    self.args.ramper_lpips_enc, 
                                                    epoch = epoch)
        self.cof_color_enc = ramper_weight_minToMax(self.args.weight_color_enc, 
                                                    self.args.ramper_color_enc, 
                                                    epoch = epoch)
        self.cof_yuv_enc = ramper_weight_minToMax(self.args.weight_yuv_enc, 
                                                    self.args.ramper_yuv_enc, 
                                                    epoch = epoch)
        self.cof_detr_en = ramper_weight_minToMax(self.args.weight_detr_enc, 
                                                    self.args.ramper_detr_enc, 
                                                    epoch = epoch)
        self.cof_lpips_croper_enc = ramper_weight_minToMax(self.args.weight_lpips_cropper, 
                                                    self.args.ramper_lpips_cropper, 
                                                    epoch = epoch)
        self.cof_riem_enc  = ramper_weight_minToMax(self.args.weight_riem_enc, 
                                                    self.args.ramper_riem_enc, 
                                                    epoch = epoch)
        self.cof_mse_enc  = ramper_weight_minToMax(self.args.weight_mse_enc, 
                                                    self.args.ramper_mse_enc, 
                                                    epoch = epoch)
    def freeze_network(self, model):
        # Freeze all parameters in DecoderNet
        for param in model.parameters():
            param.requires_grad = False    
    def unfreeze_network(self, model):
        # Freeze all parameters in DecoderNet
        for param in model.parameters():
            param.requires_grad = True    

###################################################################################################

def ramper_weight_minToMax(weight, ramper, epoch):
    cof = np.array([min(weight * epoch / ramper, 
                       weight)]).astype("float32")[0]
    return cof

###################################################################################################
def normalize_fixed(x, current_range, normed_range):
    # current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
    current_min, current_max = current_range[0], current_range[1]

    # normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
    normed_min, normed_max = normed_range[0], normed_range[1]

    x_normed = (x - current_min) / (current_max - current_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    return x_normed
###################################################################################################  
def detect_nan_hook(grad,):
    if torch.isnan(grad):#.any():
        print(tcolors.RED,
        f"NaN detected in gradients in network!"
              , tcolors.ENDC)
    return grad

def check_loss_nan(loss, name=""):
    if torch.isnan(loss):
        print(f"NaN detected in {name} loss")