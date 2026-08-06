




#######################################################################
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import tensorflow as tf
# import wandb

import keras
import numpy as np
import cv2
import sys
# from Networks_lib.Network import define_varibles_to_display
from FarhadCV.Tools import tcolors, bcolors, estimator, read_files
from FarhadCV.Tools import mkdirfile
import argparse
from Tools_nvidia_torch.torch_utils import check_free_space
#######################################################################
import warnings
tf.get_logger().setLevel('DEBUG')
warnings.filterwarnings('ignore')
#######################################################################




#######################################################################
###   Custom-Tensorboard-Callback for live display  during trainig  ###
#######################################################################
class Live_Monitoring(keras.callbacks.Callback):
    def __init__(self, myargs,  
                 args_noise , log_dir:str, 
                 display_interval:int, 
                 devices:list,
                 gpu_id,
                 world_size,):
        del_file(log_dir)

        self.myargs         = myargs
        self.args_noise     = args_noise
        self.summary_writer = tf.summary.create_file_writer(
                                            log_dir + self.myargs.name_model, 
                                            filename_suffix=self.myargs.name_model)
        self.display_interval = display_interval
        self.devices          = devices 
        self.batch            = myargs.initial_epoch  
        self.gpu_id           = gpu_id
        self.world_size       = world_size

    def on_train_begin(self, logs=None):
        #  sys.stdout.write('\r'+ run)
        print()
        
        print(tcolors.BLUE + bcolors.WHITE, 'Apply-Noise:', tcolors.RED, self.myargs.apply_noise, tcolors.ENDC)
        print(tcolors.BLUE + bcolors.WHITE, 'Apply-Warper:', tcolors.RED, self.myargs.apply_warper, tcolors.ENDC)
        if self.myargs.apply_noise:
            print(tcolors.BOLD+bcolors.BLACK, 
              "Applying Blur, Contrast, Brightness, Hue, JPEG, Gassian, and Desaturation augmentations", 
              tcolors.ENDC)
        if self.myargs.apply_warper:
            print(bcolors.BLACK+tcolors.WHITE+tcolors.BOLD,
            f"Adding Warping Version 2 - RAMP:{self.args_noise.rnd_rotation_ramp}"+
            f", max_rotate:{self.args_noise.max_rotation}"
            + tcolors.ENDC) 
        print(tcolors.BLUE + bcolors.WHITE, 'BATCH-SIZE For every GPU:', tcolors.RED, 
              self.myargs.batch_size, tcolors.ENDC)
        print(tcolors.RED + bcolors.BLACK + tcolors.BOLD, 
              'TOTAL BATCH-SIZE:', tcolors.RED, 
              self.myargs.batch_size*self.world_size, tcolors.ENDC)
        if self.myargs.use_validation:
            print(tcolors.BOLD+tcolors.BLUE + bcolors.WHITE, 
                  'BATCH-VAL:',
                  tcolors.RED, self.myargs.batch_val, 
                  tcolors.ENDC)
        print(tcolors.BLUE + bcolors.WHITE, 'STEP-PER-EPOCHS:', self.myargs.steps_per_epoch, tcolors.ENDC)
        print(tcolors.BLUE + bcolors.WHITE, 'EPOCHS:', self.myargs.epochs, tcolors.ENDC)
        print(tcolors.BLUE + bcolors.WHITE, 'Steps:', self.myargs.epochs * self.myargs.steps_per_epoch, tcolors.ENDC)

        print(tcolors.BLUE + bcolors.WHITE, 'Image-Size:', self.myargs.image_size, tcolors.ENDC)
        print(tcolors.BLUE + bcolors.WHITE, 'Message-Size:', self.myargs.secret_size, tcolors.ENDC)
        print(tcolors.BLUE + bcolors.WHITE, 'GPU-NUM:', self.myargs.gpu_devices, tcolors.ENDC)
        print(tcolors.BLUE + bcolors.WHITE, 
              'GPU-ID (which GPU this process is running on.):', self.gpu_id, 
              tcolors.ENDC)
        print(tcolors.BLUE + bcolors.WHITE, 
              'world_size (number of GPUs you want to use):', self.world_size, 
              tcolors.ENDC)
        print(tcolors.BLUE + bcolors.WHITE, 'Load-PreModels:', self.myargs.load_mode, tcolors.ENDC)

        # print(tcolors.BLUE+bcolors.WHITE, "trainable-VARS:", len(self.myargs.trainable_vars), tcolors.ENDC)
        # print(tcolors.BLUE+bcolors.WHITE, 'Model-VARS: %.2f MB' % (self.myargs.size_model), tcolors.ENDC) 
        print(tcolors.BOLD+tcolors.BLUE + bcolors.WHITE, 'type_Covarianceloss encoder:', 
              tcolors.BLUE, self.myargs.type_covarianceloss_encoder, tcolors.ENDC)
        print(tcolors.BOLD+tcolors.BLUE + bcolors.WHITE, 'type_Covarianceloss decoder:', 
              tcolors.BLUE, self.myargs.type_covarianceloss_decoder, tcolors.ENDC)
        print(tcolors.BOLD+tcolors.BLUE + bcolors.WHITE, 'Name-Model:', 
              tcolors.RED, self.myargs.name_model, tcolors.ENDC) 
   

    def on_train_batch_end(self, epoch, gpu_id, logs=None, logs2=None):

        # print(tcolors.RED, "logs", logs, tcolors.ENDC)
        

        if (epoch == 0 or epoch == 10 
            or epoch == self.myargs.initial_epoch):
            self.on_train_begin()

        run1 = (bcolors.BLACK + tcolors.BOLD+tcolors.GREEN+f"[STEPS: {epoch}/{self.myargs.epochs*self.myargs.steps_per_epoch}  - ")
        run2 = f"[[GPU{gpu_id}] - total_loss:{logs['total_loss'].cpu().detach().numpy()}]"
        # + " - mae: "+ str(logs['mae']
        # run3 =  ('- time_noisy: '+ str(self.model.VAR_DISPLAY.timesteps_var.numpy()[:2] )+"  ]   "+ tcolors.ENDC) #(tcolors.BLUE  + f"- NAME-M: JT3]" + tcolors.ENDC) 
        # run3 = (str(self.model.VAR_DISPLAY.transform_factor[1].numpy())
        #         + str(self.model.VAR_DISPLAY.transform_factor[2].numpy())
        #         + tcolors.ENDC)
        sys.stdout.write('\r'+ str(run1)+ str(run2) + tcolors.ENDC)#+ str(run3))
        # print('time_noisy: ', self.model.VAR_DISPLAY.timesteps_var)

        try:
     
            assert check_free_space("./")["Free Space (GB)"] > 0.1
            self.display_scalars(
                    {  
                    ######-----------------------------------------------------------------
                    "TRAIN0/01_total_loss":    logs["total_loss"].cpu().detach().numpy(),  
                    "TRAIN0/02_encoder_loss":  logs["encoder_loss"].cpu().detach().numpy(),
                    "TRAIN0/03_decoder_loss":  logs["decoder_loss"].cpu().detach().numpy(),
                    "TRAIN0/04_bit_acc":       logs["bit_acc"].cpu().detach().numpy(),
                    "TRAIN0/05_str_acc":       logs["str_acc"].cpu().detach().numpy(),
                    # "TRAIN0/06_lpips_pattern": logs["lpips_pattern"].cpu().detach().numpy(),

                    # "TRAIN0/06_affine_loss_en": logs["affine_loss_en"].cpu().detach().numpy(),
                    # "TRAIN0/07_affine_loss_de": logs["affine_loss_de"].cpu().detach().numpy(),
                    ######-----------------------------------------------------------------
                    "TRAIN1/01_G_loss":   logs["G_loss"].cpu().detach().numpy(),
                    "TRAIN1/02_D_loss":   logs["D_loss"].cpu().detach().numpy(),
                    "TRAIN1/03_lpips_en": logs["lpips_en"].cpu().detach().numpy(),
                    "TRAIN1/04_color_en": logs["color_en"].cpu().detach().numpy(),
                    # "TRAIN1/05_qs_en":    logs["qs_en"].cpu().detach().numpy(),
                    # "TRAIN1/06_vgg_en":   logs["vgg_en"].cpu().detach().numpy(),
                    "TRAIN1/07_mse_en":   logs["mse_en"].cpu().detach().numpy(),
                    "TRAIN1/08_yuv_loss_en":   logs["yuv_en"].cpu().detach().numpy(),
                    "TRAIN1/09_covariance_en": logs["covariance_en"].cpu().detach().numpy(),


                    ######-----------------------------------------------------------------
                    "TRAIN2/01_bce_loss":      logs["bce_loss"].cpu().detach().numpy(),
                    "TRAIN2/02_covariance_de": logs["covariance_de"].cpu().detach().numpy(),
                    



                    


                    # "TRAIN5_lossramper/00_lpips_en_loss_scale": self.model.VAR_DISPLAY.lpips_en_loss_scale[0],
                    # "TRAIN5_lossramper/01_G_loss_scale":        self.model.VAR_DISPLAY.G_loss_scale[0],
                    # "TRAIN5_lossramper/02_color_en_loss_scale": self.model.VAR_DISPLAY.color_en_loss_scale[0],
                    # "TRAIN5_lossramper/03_vgg_loss_scale":      self.model.VAR_DISPLAY.vgg_loss_scale[0],
                    # "TRAIN5_lossramper/04_qsattn_de_loss_scale":self.model.VAR_DISPLAY.qsattn_de_loss_scale[0],
                
                    }
                    , epoch)
            # if self.myargs.apply_noise:
            #     self.display_scalars({
            #         "TRAIN4_argument/01_jpeg_quality":   logs["jpeg_quality"],
            #         "TRAIN4_argument/03_blur_strength":  logs["blur_strength"],
            #         "TRAIN4_argument/05_min_contrast_scale": logs["min_contrast_scale"],
            #         "TRAIN4_argument/06_max_contrast_scale": logs["max_contrast_scale"],
            #         "TRAIN4_argument/07_min_brightness":     logs["min_brightness"],
            #         "TRAIN4_argument/08_max_brightness":     logs["max_brightness"],
            #         "TRAIN4_argument/09_rnd_min_saturation": logs["rnd_min_saturation"],
            #         "TRAIN4_argument/10_rnd_max_saturation": logs["rnd_max_saturation"],
            #         "TRAIN4_argument/11_factor_Sharpness":   logs["factor_Sharpness"],
            #         }
            #         , epoch+self.batch)

        except Exception as e:

            print(tcolors.RED,"The drive is full (104)!!",tcolors.ENDC)    
            print(tcolors.RED, "An error occurred:", e,tcolors.ENDC)


 
     
        if (epoch)%self.display_interval == 0 :
    
            for key in logs2.keys():
                try:
                    if key is not "cover_batch":
                        # logs2[key] = normalize_fixed(logs2[key], 
                        #                             current_range=[-1,1], 
                        #                             normed_range=[0,1])
                        # print(tcolors.RED,"key:", key,tcolors.ENDC)
                        logs2[key] = logs2[key][:,:,:,:].cpu().detach().permute(0, 2, 3, 1).numpy()
                    else:
                        logs2[key] = logs2[key][:,:,:,:].cpu().detach().permute(0, 2, 3, 1).numpy()
                except Exception as e:
                    pass
                    # print(tcolors.RED,"The drive is full (102)!!",tcolors.ENDC)
                    # print(tcolors.RED,"key: ", key,tcolors.ENDC)

                    # print("An error occurred:", e)
        # if True:
            try:
                assert check_free_space("./")["Free Space (GB)"] > 0.1
                self.display_images(
                    {   
                        ######-----------------------------------------------------------------
                        "00_IMG/001_cover_batch":     logs2["cover_batch"][:1,:,:,:],
                        "00_IMG/002_encoded_image":   logs2["encoded_image"][:1,:,:,:],
                        "00_IMG/003_recover_message": logs2["recover_message"][:1,:,:,:],
                        ######-----------------------------------------------------------------
                        "01_IMG/001_cover_batch":    logs2["cover_batch"][:1,:,:,:],
                        "01_IMG/002_embedding":      logs2["embedding"][:1,:,:,:],
                        "01_IMG/003_code_input_de":     logs2["code_batch_de"][:1,:,:,:],
                        "01_IMG/004_code_input_en":     logs2["code_batch_en"][:1,:,:,:],
                        # "01_IMG/005_paterned_image": logs2["paterned_image"][:1,:,:,:],
                        # "01_IMG/006_flow_v":         logs2["flow_v"][:1,:,:,:],
                        # "01_IMG/007_flow_u":         logs2["flow_u"][:1,:,:,:],
                         

                        ######-----------------------------------------------------------------
                        "02_IMG/001_cover_warper":      logs2["cover_warper"][:1,:,:,:],
                        # "01_IMG/002_encoded_image":   logs2["encoded_image"][:,:,:,:],
                        "02_IMG/003_resduial":          logs2["resduial"][:1,:,:,:],
                        "02_IMG/004_augmented_encoded": logs2["augmented_encoded"][:1,:,:,:],
                        "02_IMG/006_warper_encoded":    logs2["warper_encoded"][:1,:,:,:],
                        "02_IMG/007_dapth":             logs2["dapth"][:1,:,:,:],
                        # "02_IMG/008_dapth_warper": logs2["dapth_warper"][:1,:,:,:],


                        # "02_IMG/008_croped_de":         logs2["croped_de"][:,:,:,:],
                        # "02_IMG/009_effine_en":         logs2["effine_en"][:,:,:,:],
                        # "02_IMG/010_cropped_mask":      logs2["cropped_mask"][:,:,:,:],



                        ######-----------------------------------------------------------------





                        ###-------------------------------------------------------------------------
                        # "TEST/0_original_image":self.model.VAR_DISPLAY.original_image_test,
                        # "TEST/1_x_t":self.model.VAR_DISPLAY.x_t_test,
                        # "TEST/2_pred_xstart_test":self.model.VAR_DISPLAY.pred_xstart_test,
                        ###-------------------------------------------------------------------------


                    }, epoch)

                    
                
                    
                    ##-----------------------------------------------------
                    ## display loss and metrics
                # self.display_scalars(
                #     {    
                
                #     "TRAIN4/04_learning_rate":                  self.model.optimizer.learning_rate,
                #     "TRAIN4/05_mean_trainable_variables_first": keras.ops.mean(self.model.trainable_weights[0]),
                #     "TRAIN4/06_mean_trainable_variables_last":  keras.ops.mean(self.model.trainable_weights[-1])
                    
                #     }
                #     , epoch+self.batch)
            except Exception as e:
                print(tcolors.RED,"The drive is full (107)!!",tcolors.ENDC)
                print("An error occurred:", e)

        # del(self.model.VAR_DISPLAY)
        # self.model.VAR_DISPLAY = define_varibles_to_display(myargs=self.myargs, 
        #                                                     image_size=256, devices=self.devices)
        del(logs2)
        del(logs)

        self.batch += 1
        self.epoch = epoch
        # del(logs)

    def on_test_end(self, logs=None):
        self.display_images(
                {
                    # "IMG/original_image":self.model.VAR_DISPLAY.original_image, 
                    # # "IMG/x_start":self.term_dic['x_start'], 
                    # "IMG/pred_xstart":self.model.VAR_DISPLAY.pred_xstart,
                    # "IMG/x_t":self.model.VAR_DISPLAY.x_t, 
                    # "NOISE/target_noise":self.model.VAR_DISPLAY.target_noise, 
                    # "NOISE/pred_noise":self.model.VAR_DISPLAY.pred_noise,
                    ###-------------------------------------------------------------------------
                    "TEST/0_original_image":self.model.VAR_DISPLAY.original_image_test,
                    "TEST/1_x_t":self.model.VAR_DISPLAY.x_t_test,
                    "TEST/2_pred_xstart_test":self.model.VAR_DISPLAY.pred_xstart_test,
                    # "TEST/3_pred_xstart_test":self.model.VAR_DISPLAY.recovered_message_test,

                    ###-------------------------------------------------------------------------


                }, self.epoch)
        self.display_scalars({
            "TEST/01_lpips": logs["lpips"],
            # "TEST/02_loss_message": logs["loss_message"],

        }, self.epoch)

    def display_images(self, dict , step):
        namelist = list(dict.keys())
        valueslist = list(dict.values())
        with self.summary_writer.as_default():
            for name, values in zip(namelist, valueslist):
                # print(tcolors.RED, name, ":", values, tcolors.ENDC)
                tf.summary.image("{}".format(name), values[:1,:,:,:3], step=step)
        del(dict)


    def display_scalars(self, dict, step):
        namelist = list(dict.keys())
        valueslist = list(dict.values())
        with self.summary_writer.as_default():
            for name, values in zip(namelist, valueslist):
                tf.summary.scalar("{}".format(name), values, step=step)
        del(dict)




#######################################################################
####### Tools:                              ##########
#######################################################################
def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)





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