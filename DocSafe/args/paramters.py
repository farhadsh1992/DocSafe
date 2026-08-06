

"""
@--03.02.2025--@
Author: github/farhadsh1992
INFO:
LAST_UPDATE:

"""

import warnings
warnings.filterwarnings('ignore')

def getArgsInputs():
    import argparse
    parser = argparse.ArgumentParser()
    ################################################################################################
    parser.add_argument('--name_model', help="", type=str, 
                default="StampOneDetr2_SIRIN5_X5010_X5C012")
    # StampOneDetr2_SIRIN5_X5010_X5B011
    parser.add_argument('--discribation',  help="", type=str, 
                        default=
                        """
                        Datatset: ffhq_20G
                        """)
    parser.add_argument('--gpu_devices',    help="select which GPU to run a job on", type=str,  
                        default="1")
    parser.add_argument('--devices_number', help="select which GPU to run a job on", type=list, 
                        default=[0, 0, -1])

    parser.add_argument('--using_fit_for_train', help="", type=bool, default = True)
    parser.add_argument('--using_stratgy',       help="", type=bool, default = False)
    parser.add_argument('--use_validation',      help="", type=bool, default = False)   
    parser.add_argument('--Just_train_decoder',  help="", type=int, default = 20000) #7000

    parser.add_argument('--apply_Stega_Discrimnator',     help="", type=bool, default = False)   
    parser.add_argument('--apply_Specteral_Discrimnator', help="", type=bool, default = False)  
    ##################################################################################################
    parser.add_argument('--type_covarianceloss_encoder', 
                        help="log-euclidean || airm || jeffrey || stein", type=str, 
                        default="airm2")
    parser.add_argument('--type_covarianceloss_decoder', 
                        help="log-euclidean2 || airm2 || jeffrey2 || stein2", type=str, 
                        default="airm2")
    parser.add_argument('--cov_discribe', 
                        help="", type=str, 
                        default="norm0t01Harmonic")
    ## log-euclidean [], airm[],  jeffrey [], stein[],
    ## log-euclidean2 [], airm2[],  jeffrey2-1 [Done], jeffrey2-2 [], stein2[],
    ################################################################################################## 
    ## Affine Setting
    parser.add_argument('--apply_detr', help="", type=bool, default = False)        
    parser.add_argument('--croper_size',       help="", type=int, default = 64)
    parser.add_argument('--mask_ramper',       help="", type=int, default = 160000)

    ##################################################################################################
    parser.add_argument('--epochs',           help="", type=int, default = 400000)
    parser.add_argument('--steps_per_epoch',  help="", type=int, default = 100)
    parser.add_argument('--batch_size',       help="", type=int, default = 4) ##8
    parser.add_argument('--batch_test',       help="", type=int, default = 1)
    parser.add_argument('--initial_epoch',    help="", type=int, default = 380000)
    ########################################################################################
    parser.add_argument('--pattern_img', help="", type=str, 
                        default="./pattern/wave_02.jpg")
    # ### SIRIN
    # type_data = "ffhq_20G" ## Fashian_COCO || ffhq_20G || Fashian || Car_Fashian || Car_dataset_train
    # parser.add_argument('--TRAIN_COVER', help="", type=str, 
    #                     default=f"/media/ssd2_data/Farhad_Sirin_ssd2/DATASETS/{type_data}/")
    # parser.add_argument('--TEST_COVER', help="", type=str, 
    #                     default=f"/media/ssd2_data/Farhad_Sirin_ssd2/DATASETS/{type_data}/")
    # parser.add_argument('--random_backgrounds_file', help="", type=str, 
    #                     default="./images/passport/")
    # parser.add_argument('--TRAIN_BOX', help="", type=str, 
    #         default="/media/ssd2_data/Farhad_Sirin_ssd2/Code_36v/0MASK_DATASET/save_text/")
    # parser.add_argument('--monocular_depth_path', help="", type=str, 
    #     default="/media/ssd2_data/Farhad_Sirin_ssd2/Code_36v/ControlNet_try2/"+
    #             f"monocular_depth_estimation3/dath_dataset/{type_data}/Monocular/")
    ## ////////////////////////////////////////////////////////////////////////////////////////////
    parser.add_argument('--TRAIN_COVER', help="", type=str, 
                        default=f"/media/ssd2_data/Farhad_Sirin_ssd2/Code_36v/ControlNet_try2"+
                        "/monocular_depth_estimation4/WB_dataset2/Car_FashianWB2/origin_35_3/")
    parser.add_argument('--TEST_COVER', help="", type=str, 
                        default=f"/media/ssd2_data/Farhad_Sirin_ssd2/Code_36v/ControlNet_try2"+
                        "/monocular_depth_estimation4/WB_dataset2/Car_FashianWB2/origin_35_3/")
    parser.add_argument('--random_backgrounds_file', help="", type=str, 
                        default="./images/passport/")
    # parser.add_argument('--TRAIN_BOX', help="", type=str, 
    #         default="/media/ssd2_data/Farhad_Sirin_ssd2/Code_36v/0MASK_DATASET/save_text/")
    parser.add_argument('--monocular_depth_path', help="", type=str, 
        default=f"/media/ssd2_data/Farhad_Sirin_ssd2/Code_36v/ControlNet_try2"+
                "/monocular_depth_estimation4/WB_dataset2/Car_FashianWB2/Newdepth2_35_4/")
    ## //////////|/|/|///////////////////////////////////////////////////////////////////////////////|/
    ### HUMER
    # parser.add_argument('--TRAIN_COVER', help="", type=str, 
    #                     default="/media/visteam/bigSSD1/Farhad6/000_Dataset/ffhq_20G/")
    # parser.add_argument('--TEST_COVER', help="", type=str, 
    #                     default="/media/visteam/bigSSD1/Farhad6/000_Dataset/ffhq_20G/")
    # parser.add_argument('--monocular_depth_path', help="", type=str, 
    #     default="/media/visteam/bigSSD1/Farhad6/006_Code/ControlNet_try2/"+
    #             "monocular_depth_estimation3/dath_dataset/ffhq_20G_2/Monocular/")

    # parser.add_argument('--monocular_depth_path', help="", type=str, 
    #     default="/media/visteam/bigSSD1/Farhad6/006_Code/ControlNet_try2/"+
    #             "monocular_depth_estimation3/dath_dataset/ffhq_20G_2/Monocular/")

    # parser.add_argument('--TRAIN_COVER', help="", type=str, 
    #                     default="/media/visteam/bigSSD1/Farhad6/0MASK_DATASET/ffhq_20G/")
    # parser.add_argument('--TEST_COVER', help="", type=str, 
    #                     default="/media/visteam/bigSSD1/Farhad6/0MASK_DATASET/ffhq_20G/")
    # # parser.add_argument('--TRAIN_MASK2', help="", type=str, 
    # #                     default="/media/visteam/bigSSD1/Farhad6/0MASK_DATASET/save_images/")
    # parser.add_argument('--TRAIN_BOX', help="", type=str, 
    #                     default="/media/visteam/bigSSD1/Farhad6/0MASK_DATASET/save_text/")
    
    ##################################################################################################
    parser.add_argument('--max_img',      help="", type=float, default=0.80)
    parser.add_argument('--ramper_blend', help="", type=int,   default=12000)
    
    ##################################################################################################
    parser.add_argument('--save_dir', help="", type=str, default="./results/")
    parser.add_argument('--log_dir',  help="", type=str, default="./results/logs/")
    parser.add_argument('--save_interval',    help="",  type=int, default=2000)
    parser.add_argument('--display_interval', help='Model checkpoint interval', type=int, default=5)

    ##################################################################################################
    parser.add_argument('--learning_rate', help="", type=float, default= 1e-4)
    parser.add_argument('--learning_rate_disc',     help="", type=float, default= 10e-6)
    
    parser.add_argument('--optim_name',            help="", type=str,   default="Adam")
    
   
    ##################################################################################################
    
    ##################################################################################################
    parser.add_argument('--image_size', help="", type=int, default=256)
    parser.add_argument('--image_size_de', help="", type=int, default=256)

    parser.add_argument('--message_shape', help="", type=int, default=(16,16))
    parser.add_argument('--secret_size', help="", type=int, default=100)
    parser.add_argument('--mess_reshape',  help = "for produce image in dataset", 
                        type = int, default = (10,10)) ##
    parser.add_argument('--mess_reshape2',  help = "for produce image in dataset", 
                        type = int, default = (10, 10)) ##
    parser.add_argument('--pad_size',    help="",  type=int, default=3)
    #########################################################################################################################
    # Color-Histogram P:
    parser.add_argument('--max_input_size_hist', help="", type=float, default=128)
    parser.add_argument('--histogram_size',      help="", type=float, default=128)
    parser.add_argument('--method_his',          help="", type=str,   default='inverse-quadratic')   
    # ########################################################################################################################
    # YUV-Loss weights:
    parser.add_argument('--l2_edge_gain', help="", type=float, default=10.0)
    #parser.add_argument('--yuv_scales_pl', help="", type=float, default=10e2)
    parser.add_argument('--y_scale', type=float, default= 1.0)
    parser.add_argument('--u_scale', type=float, default= 2.0)
    parser.add_argument('--v_scale', type=float, default= 2.0)

    #########################################################################################################################
    parser.add_argument('--weight_enc', help="", type=float, default=1)
    parser.add_argument('--weight_lpips_enc', help="", type=float, default=7)
    parser.add_argument('--weight_color_enc', help="", type=float, default=8)
    parser.add_argument('--weight_yuv_enc', help="", type=float, default=2)
    parser.add_argument('--weight_lpips_cropper',  help="", type=float, default=8)
    parser.add_argument('--weight_riem_enc',  help="", type=float, default=3)
    parser.add_argument('--weight_mse_enc',  help="", type=float, default=4)



    parser.add_argument('--ramper_enc',       help="", type=int, default=20000)
    parser.add_argument('--ramper_lpips_enc', help="", type=int, default=90000)
    parser.add_argument('--ramper_color_enc', help="", type=int, default=90000)
    parser.add_argument('--ramper_yuv_enc',       help="", type=int, default=20000)
    parser.add_argument('--ramper_lpips_cropper', help="", type=int, default=40000)
    parser.add_argument('--ramper_riem_enc',  help="", type=int, default=40000)
    parser.add_argument('--ramper_mse_enc',  help="", type=int, default=40000)


    #########################################################################################################################
    parser.add_argument('--weight_rec',                 help="", type=float, default=1)
    parser.add_argument('--weight_bce_rec',             help="", type=float, default=15)
    parser.add_argument('--weight_riem_rec',            help="", type=float, default=10)

    parser.add_argument('--weight_lpips_rec',           help="", type=float, default=0)
    parser.add_argument('--weight_QRCode_simulate_rec', help="", type=float, default=0) 
    parser.add_argument('--weight_seamlessCloning_rec', help="", type=float, default=0)
    parser.add_argument('--weight_spectral_rec',        help="", type=float, default=2)

    

    parser.add_argument('--ramper_rec',          help="", type=int, default=20000)
    parser.add_argument('--ramper_bce_rec',      help="", type=int, default=20000)
    parser.add_argument('--ramper_riem_rec',     help="", type=int, default=20000)
    parser.add_argument('--ramper_lpips_rec',    help="", type=int, default=20000)
    parser.add_argument('--ramper_spectral_rec', help="", type=int, default=1)
    #######################################################################################
    parser.add_argument('--weight_detr_enc', help="", type=float, default = 2)
    parser.add_argument('--ramper_detr_enc', help="", type=int,   default = 20000)
    parser.add_argument('--weight_detr_rec', help="", type=float, default = 2)
    parser.add_argument('--ramper_detr_rec', help="", type=int,   default = 20000)
    # *****************************************************************************************
    # For QR_Puncher_simulate puncherNet:
    parser.add_argument('--Dis_b', help="", type=float, default=70)
    parser.add_argument('--Dis_w', help="", type=float, default=180)
    parser.add_argument('--Correct_b', help="", type=float, default=40)
    parser.add_argument('--Correct_w', help="", type=float, default=220)
    parser.add_argument('--thershold', help="", type=float, default=127.5)

    # *****************************************************************************************
    # For QRCode_simulate RecoverNet:
    parser.add_argument('--Dis_b_rec', help="", type=float, default=50)
    parser.add_argument('--Dis_w_rec', help="", type=float, default=200)
    parser.add_argument('--Correct_b_rec', help="", type=float, default=0)
    parser.add_argument('--Correct_w_rec', help="", type=float, default=225)
    parser.add_argument('--thershold_rec', help="", type=float, default=127.5)
   

    # *****************************************************************************************
    # weights_RecoverNet***********************************************************************
    # For colorhistogram:
    parser.add_argument('--max_input_size_hist_forrecovernet', help="", type=float, default=512)
    parser.add_argument('--histogram_size_forrecovernet', help="", type=float, default=64)

    ########################################################################################################################
    parser.add_argument('--apply_noise', \
                        help = "apply physical and printer noise to moive printer-proof", 
                        type = bool, default = True) 
    parser.add_argument('--apply_warper', \
                        help = "apply warper", 
                        type = bool, default = True)  # False || True
    
    parser.add_argument('--apply_passport_noise', \
                        help = "simulate background of passport", 
                        type = bool, default = False) 
    parser.add_argument('--apply_dithering_noise', \
                        help = "sapply dithering", 
                        type = bool, default = False) 
    
    ########################################################################################################################
    parser.add_argument('--initialize_detr', 
                        help = "[False, or True]", 
                        type = bool, default = False)
    name_model = "StampOneDetr_SIRIN_CompleteFFT_X002"
    steps = str(70000)
    parser.add_argument('--path_detr_weights1', help="", type=str, 
        default= f"./results/results_final/{name_model}/ckpts/"+
        f'Steps{steps}_DetrNet_{name_model}.pt')
    ########################################################################################################################
    # ########################################################################################################################
    """ Load Pre-trained models: """
    parser.add_argument('--load_mode', 
                        help = "[False, or True]", 
                        type = bool, default = False)
    
    name_model = "StampOneDetr2_SIRIN5_X5010_X5C012"
    steps = str(380000)
    ## StampOne_TF_STN_v205_T5012 ## 170000 ## 110000
    ## StampOne_TF_STN_swinde_v205_T5031 ## 20000 ## 
    ## StampOne_TF_STN_mixerAll_snake_v207_T7116 # 70000
    
    parser.add_argument('--path_encoder_weights', help="", type=str, 
        default= f"./results/results_final/{name_model}/ckpts/"+
        f'Steps{steps}_EncoderNet_{name_model}.pt')
    parser.add_argument('--path_decoder_weights', help="", type=str, 
        default= f"./results/results_final/{name_model}/ckpts/"+
        f'Steps{steps}_DecoderNet_{name_model}.pt')    
    parser.add_argument('--path_stegadisc_weights', help="", type=str, 
        default= f"./results/results_final/{name_model}/ckpts/"+
        f'Steps{steps}_StegaDisc_{name_model}.pt')
    parser.add_argument('--path_spectraldisc_weights', help="", type=str, 
        default= f"./results/results_final/{name_model}/ckpts/"+
        f'Steps{steps}_SpectralDisc_{name_model}.pt')
    
    



    args = parser.parse_args()
    return args
    