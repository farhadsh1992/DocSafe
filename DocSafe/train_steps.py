

"""
@--03.02.2025--@
Author: github/farhadsh1992
INFO:
    -ref: 
        
    
LAST_UPDATE:
"""

####################################################################################
import os 
# Enable CUDA Device-Side Assertions for debugging
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import time
import pickle
import pandas as pd
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import torch.multiprocessing as mp
from .args.paramters import getArgsInputs
args = getArgsInputs()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
# This guide can only be run with the torch backend.
os.environ["KERAS_BACKEND"] = "torch"
####################################################################################
from FarhadCV.Tools import tcolors, bcolors, estimator, read_files
from FarhadCV.Tools import mkdirfile
from FarhadCV.Tools import copy_files, copy_folder
from FarhadCV.Tools_send_notifaction import Message_Sender
####################################################################################
# DNN Libs:
import torch
import keras
# from torchvision import transforms
# from torch.utils.data import DataLoader
####################################################################################
from Tools_nvidia_torch.torch_utils import  Configure_GPU, CHECK_PYTHON_SETTING
###########################################################################
##   Pin GPU to be used to process local rank (one GPU per process)     ##
##########################################################################
# CHECK_PYTHON_SETTING()


devices = Configure_GPU(args)

##############################################
##                   MAIN                   ##
##############################################
##############################################
##                   MAIN                   ##
##############################################
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "14501" ## port must have value from 0 to 65535
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def  main_test(rank: int, world_size: int):

    # destroy_process_group()
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
    # Define paramters (SETTING)
    args = getArgsInputs()
    from .args.paramters_noise import getArgsInputsNoise
    args_noise = getArgsInputsNoise()
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
    
    ###################################################################################
    copy_files(source="./paramters.py", 
               distincation="./results/results_final/"+ args.name_model +"/"+f"paramters_{args.name_model}.py")
    copy_files(source="./CustomFit.py", 
               distincation="./results/results_final/"+ args.name_model +"/"+f"CustomFit_{args.name_model}.py")
    copy_files(source="./paramters_noise.py", 
        distincation="./results/results_final/" + args.name_model +"/"+f"paramters_noise_{args.name_model}.py")
    copy_files(source="./paramters_detr.py", 
        distincation="./results/results_final/" + args.name_model +"/"+f"paramters_detr_{args.name_model}.py")
    copy_files(source="./train_steps.py", 
               distincation="./results/results_final/"+ args.name_model +"/"+f"train_steps_{args.name_model}.py")
    
    ####### 
    # copy_folder(src_folder="./Network_Libs/", 
    #             dest_folder="./results/results_final/"+ args.name_model +"/Files/Network_Libs/")
    # copy_folder(src_folder="./NoiseSimulationPrime_Torch", 
    #             dest_folder="./results/results_final/"+ args.name_model +"/Files/NoiseSimulationPrime_Torch/")
    

    ##############################################
    ##      LIVE MONITORS RUNNING STATUS        ##
    ##############################################
    from .monitors.custom_callback import Live_Monitoring
    Live_Monitors = Live_Monitoring(myargs  = args, 
                                    args_noise = args_noise,
                                    log_dir = args.save_dir + "logs/", 
                                    display_interval = args.display_interval, 
                                    gpu_id  = rank, # which GPU this process is running on.
                                    world_size = world_size, # number of GPUs you want to use
                                    devices = device_one)
    
    
    Message_Sender_router = Message_Sender()

    ##########################################################################
    ######           Load Image Augmentation and Warper                ######
    ##########################################################################
    from .augmentors.Augmentor_v01 import Augmentation_Transformer
    augmentor = Augmentation_Transformer(
                                    args       = args_noise, 
                                    batch_size = args.batch_size, 
                                    image_size = args.image_size, 
                                    device     =device_one )
    
    from .augmentors.warper_Augmentor_v01 import Warper_Transformer
    warper_router = Warper_Transformer(
                                args       = args, 
                                noise_args = args_noise, 
                                batch_size = args.batch_size,  
                                image_size = args.image_size,  
                                device     = device_one)
    ##########################################################################
    ######             Load Dataset                            ######
    ##########################################################################
    # from datasets_lib.Loaddataset_v1 import Dataset_Router
    # from datasets_lib.Loaddataset_v2 import Dataset_Router
    # from datasets_lib.Loaddataset_v3 import Dataset_Router
    # from datasets_lib.Loaddataset_v4 import Dataset_Router
    # from datasets_lib.Loaddataset_v6 import Dataset_Router
    from .datasets.Loaddataset_v7 import Dataset_Router


    # if args.using_pad_max:
    # from datasets_lib.Loaddataset_v5 import Dataset_Router
    datasetOS_train, datasetOS_test = Dataset_Router(args = args, 
                                            TRAIN_COVER = args.TRAIN_COVER,
                                            TEST_COVER = args.TEST_COVER,
                                            rank = rank, # which GPU this process is running on.
                                            world_size = world_size, # number of GPUs you want to use
                                            device = device_one).load()

    ############################################################################
    ##           Choose the suitable CustomFit for extraction                  ##
    ############################################################################
    from .CustomFit import Trainer as CustomFit_StampOne2
    SO2_router = CustomFit_StampOne2(
                                args       = args, 
                                args_noise = args_noise,
                                train_data = datasetOS_train,
                                devices    = device_one)
    SO2_router.compile()
    SO2_router.upload_live_monitor(Live_Monitors = Live_Monitors)
    SO2_router.upload_augmentation(augmentor     = augmentor,
                                  warper_router  = warper_router)


    ############################################################################
    ##                                                                        ##
    ############################################################################
    if world_size == 1:
        SO2_router.train_with_one_gpu(args.epochs)
    elif world_size > 1:
        SO2_router.train_multi_gpus(args.epochs)
        destroy_process_group()

    ############################################################################
    ##                                                                        ##
    ############################################################################
##############################################
##                   MAIN                   ##
##############################################
if __name__ == "__main__":
    start_time = time.time()
    devices = Configure_GPU(args)
    world_size = torch.cuda.device_count()
    # devices = 0
    print("world_size",world_size)
    if world_size == 0:
        main_test(world_size, world_size)
    if world_size == 1:
        main_test(world_size, world_size)
    else:
        mp.spawn( main_test, 
                args=(world_size,), 
                nprocs=world_size)


    end_time = time.time()

    print()
    print("time of training:", str(end_time - start_time), " seconds")
    print("<<<-----|> IT IS DONE <|----->>")