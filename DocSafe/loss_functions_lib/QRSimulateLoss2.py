

import torch
from torch import nn
from Network_Libs.ArtCoder.SS_layer import SSlayer
from Network_Libs.ArtCoder.tools import rgb_to_grayscale

class primery_QR_code_simulatorFast(torch.nn.Module):
    def __init__(self, batch, 
                 model_size, 
                 model_num, thershold, 
                 Dis_b, Dis_w, Correct_b, Correct_w,
                 devices):
        super(primery_QR_code_simulatorFast, self).__init__()


        self.batch      = batch
        self.model_size = model_size
        self.model_num = model_num
        self.Dis_b     = (Dis_b / 127.5) - 1
        self.Dis_w     = (Dis_w / 127.5) - 1
        self.Correct_b = (Correct_b)# - 1
        self.Correct_w = Correct_w #- 1
        self.thershold = (thershold / 127.5) - 1

        self.devices = devices

        self.ss_layer_extractor = SSlayer(model_size=model_size, 
                                          requires_grad=False).to(self.devices[0])
        self.USE_ACTIVATION_MECHANISM = True

        self.loss_MSE = nn.MSELoss()

        # Example input: img_code is (B, C, H, W)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=(self.model_size, self.model_size))
        self.avg_pool2 = torch.nn.AvgPool2d(kernel_size=(self.model_size, self.model_size))

    def get_action_matrix(self, img_target, img_code):
        # Example input: Assuming img_target and img_code are (B, C, H, W) in PyTorch
        img_target = rgb_to_grayscale(img_target)
        img_code = rgb_to_grayscale(img_code)

        # Convert to float32 (PyTorch defaults to float32, but explicit casting is good)
        img_target = img_target.float()
        img_code   = img_code.float()

        ideal_result = self.get_binary_result(img_code)
        center_mat   = self.get_center_pixel(img_target)
        error_module = self.get_error_module(center_mat, ideal_result)

        return error_module, ideal_result
    def get_binary_result(self, img_code):
        # Apply average pooling
        module = self.avg_pool(img_code)

        # Apply thresholding conditions
        out = torch.where(module < self.thershold, module, torch.tensor(1.0, device=module.device))
        binary_result = torch.where(module > self.thershold, out, torch.tensor(0.0, device=module.device))
        return binary_result

    def get_target(self, binary_result):
        # Calculate image size
        img_size = self.model_size * self.model_num 

        # Initialize the target array (equivalent to np.require with uint8)
        target = torch.ones((img_size, img_size), dtype=torch.uint8, device=binary_result.device) 

        # Populate the target matrix
        for i in range(self.model_num):
            for j in range(self.model_num):
                one_binary_result = binary_result[i, j]
                if one_binary_result == 0:
                    target[i * self.model_size:(i + 1) * self.model_size, j * self.model_size:(j + 1) * self.model_size] = self.Correct_b
                else:
                    target[i * self.model_size:(i + 1) * self.model_size, j * self.model_size:(j + 1) * self.model_size] = self.Correct_w

        # Convert target to float32 (equivalent to tf.cast)
        target = target.float()

        # Expand dimensions at axis 2 (equivalent to tf.expand_dims(axis=2))
        target = target.unsqueeze(1)  # Shape: (H, W, 1)

        # Convert grayscale to RGB (equivalent to tf.image.grayscale_to_rgb)
        target = target.expand(3, -1, -1)  # Shape: (3, H, W)

        # Normalize to range [-1, 1] (instead of [0, 1])
        target = (target / 127.5) - 1.0
        return target
    def forward(self):
        pass
    def get_center_pixel(self, img_target):
        center_mat = self.avg_pool2(img_target)
        return center_mat
    def get_error_module(self, center_mat, code_result):
        # Create an error_module1 tensor filled with zeros
        error_module1 = torch.zeros((self.batch, self.model_num, self.model_num, 1), 
                                    device=center_mat.device)
        
        # Reference variables
        center_pixel = center_mat
        right_result = code_result

        # Apply conditions using torch.where
        error_module = torch.where(((right_result == 0) & (center_pixel < self.Dis_b)), error_module1, 
                                   torch.tensor(1.0, device=center_mat.device))
        error_module = torch.where(((right_result == 1) & (center_pixel > self.Dis_w)), error_module, 
                                   torch.tensor(1.0, device=center_mat.device))

        # Delete unused variable (optional in PyTorch, as memory is managed automatically)
        del(error_module1) 
        return error_module
    def get_inputs(self, image_patern, code_image ):
        self.code_image = code_image

        self.ideal_Feature_Map = self.ss_layer_extractor(code_image)
    def get_labels(self, images, code_image):
        self.ideal_Feature_Map = self.ss_layer_extractor(code_image)
        self.Feature_Map_target = self.ss_layer_extractor(images)

        if self.USE_ACTIVATION_MECHANISM :
            error_matrix, binary_result = self.get_action_matrix(img_target = images, # ???>tensor_to_PIL(target_image)
                                                                img_code    = code_image )
            


            # Sum of error_matrix
            activate_num = torch.sum(error_matrix)

            # Convert data type to float32
            activate_weight = error_matrix.float()

            # Compute mean along axis 3 (equivalent to TensorFlow axis=3)
            activate_weight = torch.mean(activate_weight, dim=1)

            # Expand dimension at axis 2 (equivalent to TensorFlow expand_dims at axis=2)
            activate_weight = activate_weight.unsqueeze(1)

            # Element-wise multiplication with feature maps
            self.Feature_Map_target = self.Feature_Map_target * activate_weight
            self.ideal_Feature_Map2 = self.ideal_Feature_Map * activate_weight 

    def compute_code_loss(self):
        # code_loss = tf.keras.losses.MeanSquaredError()( self.ideal_Feature_Map2, self.Feature_Map_target)
        code_loss = self.loss_MSE(self.ideal_Feature_Map2, self.Feature_Map_target)
        # del(self.ideal_Feature_Map2)
        # del(self.Feature_Map_target)
        return code_loss
    

##########################################################################
####                                             ####
##########################################################################
def MaxMinNormalization(img, maxvalue, minvalue):
    # maxvalue = np.max(loss_img)
    # minvalue = np.min(loss_img)
    img = (img - minvalue) / (maxvalue - minvalue)
    # img = np.around(img, decimals=2)
    return img