

"""
@--01.04.2026--@
Author: github/farhadsh1992
INFO:
LAST_UPDATE:
"""



from DocSafe import encoder
from DocSafe import decoder
from FarhadCV.Tools import tcolors, bcolors



if __name__ == "__main__":

    #######################################################################################
    #####       ENCODED IMAGES                                                     #####
    #####                                                                          #####
    #######################################################################################

    encoder_router = encoder(
                            model="M1", 
                            path_model="./pre_trained_models/",
                            secret_size=100,
                            BCH_BITS       = 7, # 13, 7, 21, 44
                            BCH_POLYNOMIAL = 137, #487#137#8219, 20023 
                            number_zeros   = 4,
                            )
    encoder_router.load_network(device="cpu")

    ## feed one image into encoder 
    ## output: encoder_router.original_image
    images      = encoder_router.read_image(path=["test_images/image_with_background_white_5.jpg"])

    ## preprocess images
    image_batch = encoder_router.preprocess_images(images)
    encoded_images = encoder_router(
                original_images = image_batch,
                messages       = ['viste'],
                mask           = None,
                
                        )

    binery_message = encoder_router.binery_messages



    #######################################################################################
    #####       DCODED  ENCODED-IMAGES                                                     #####
    #####                                                                          #####
    #######################################################################################
    decoder_router = decoder(
                            model="M1",  ## M1, M2, M3
                            path_model="./pre_trained_models/", 
                            secret_size=100,
                            BCH_BITS       = 7, # 13, 7, 21, 44
                            BCH_POLYNOMIAL = 137, #487#137#8219, 20023 
                            number_zeros   = 4,
                            )
    decoder_router.load_network(device="cpu")

    ## feed one image into decoder 
    ## output: decoder_router.decoded_message
    # encoded_images = decoder_router.read_image(path=["test_images/encoded_images.jpg"])
    ## or
    # encoded_images = decoder_router.get_image(encoded_images)

   
    ## preprocess encoded images
    encoded_batch  = decoder_router.preprocess_images([encoded_images])

    decoded_messages = decoder_router(
                encoded_images = encoded_batch,
                mask           = None,
                        )
    
    decoded_binery_messages = decoder_router.decoded_binery_messages
    # validate = decoder_router.validate(decoded_binery_messages, binery_message)


    print(tcolors.GREEN, "decoded_messages: ", decoded_messages, tcolors.ENDC)