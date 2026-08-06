


"""
@--03.02.2025--@
Author: github/farhadsh1992
INFO:
LAST_UPDATE:

"""


import warnings
warnings.filterwarnings('ignore')

def getArgsInputsNoise():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_model', help="", type=str, 
                default="FSetag_torch")
    
    ##############################################################
    ### Border+Warping : ###
    ##############################################################
    parser.add_argument('--borders',  help = "", type = str, 
                            choices = ['no_edge', 'image'], default = "image" )
    parser.add_argument('--speed_rotation',      help = "ss used 0.1",   type = float, 
                        default = 1)
    parser.add_argument('--max_rotation',      help = "ss used 0.1",   type = float, 
                        default = 0.08)
    parser.add_argument('--rnd_rotation_ramp', help = "ss used 10000", type = int,   
                        default = 40000)

    ##############################################################
    ###  ###
    ##############################################################
    """ JPEGS Nosies Weights and Ramps: """
    parser.add_argument('--jpeg_quality_ramp', type=float, default=5000)
    parser.add_argument('--jpeg_quality',      type=float, default=30)
    ##############################################################
    ###  ###
    ##############################################################
    """ Dithering and palette: """
    parser.add_argument('--rnd_dithering_min', type=float, default= 14)
    parser.add_argument('--rnd_dithering_max', type=float, default= 28)
    parser.add_argument('--dithering_delta',   type=float, default= 0.3)
    parser.add_argument('--dithering_ramp',    type=float, default= 10000) #10000

    ##############################################################
    ###  ###
    ##############################################################
    parser.add_argument('--rnd_palette_min',   type=float, default= 14)
    parser.add_argument('--rnd_palette_max',   type=float, default= 30)
    parser.add_argument('--palette_delta',     type=float, default= 0.3)
    parser.add_argument('--palette_ramp',      type=float, default= 20000) #20000

    ##############################################################
    ###  ###
    ##############################################################
    """ Nosies Weights: """
    parser.add_argument('--rnd_bri',       type=float, default= .2)
    parser.add_argument('--rnd_noise',     type=float, default= .01)
    parser.add_argument('--rnd_sat',       type=float, default= 1.1)
    parser.add_argument('--rnd_hue',       type=float, default= .1)
    parser.add_argument('--contrast_low',  type=float, default= .5)
    parser.add_argument('--contrast_high', type=float, default= 1.5)
    ##############################################################
    ###  Nosies Ramps: ###
    ##############################################################
    parser.add_argument('--rnd_bri_ramp',   type=int, default= 3000) #3000
    parser.add_argument('--rnd_sat_ramp',   type=int, default= 3000) #3000
    parser.add_argument('--rnd_hue_ramp',   type=int, default= 3000) #3000
    parser.add_argument('--rnd_noise_ramp', type=int, default= 3000) #3000
    parser.add_argument('--contrast_ramp',  type=int, default= 3000) #3000

    parser.add_argument('--ramp_noise3',    type=int, default= 5000) #5000
    ##############################################################
    ###  ###
    ##############################################################
    parser.add_argument('--ramp_ditering',    type=int, default= 5000) #5000
    parser.add_argument('--max_ditering',    type=int, default= 0.2) #5000


    args = parser.parse_args()
    return args




