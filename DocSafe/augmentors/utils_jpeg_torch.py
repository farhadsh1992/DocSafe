






import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np



###################################################################
####                                                  ####
###################################################################
class jpeg_compress_decompress():
    def __init__(self, 
                 downsample_c=True, 
                 rounding=None, 
                 height:int = 256, 
                 width:int = 256,
                 device=None):

        self.device = device
        self.downsample_c = downsample_c
        self.rounding = rounding

        self.height = height
        self.width  = width

    def call(self, image, factor=1):
        image *= 255


        # orig_height, orig_width = height, width
        orig_height, orig_width = image.shape[2], image.shape[3]

        ###################################################################
        ####                                                  ####
        ###################################################################
        # Check if dimensions are not multiples of 16
        if self.height % 16 != 0 or self.width % 16 != 0:
            # Round up to the next multiple of 16
            height = ((height - 1) // 16 + 1) * 16
            width = ((width - 1) // 16 + 1) * 16

            vpad = height - orig_height
            wpad = width - orig_width
            top = vpad // 2
            bottom = vpad - top
            left = wpad // 2
            right = wpad - left

            # Apply symmetric (reflect) padding
            image = F.pad(image, (left, right, top, bottom), mode='reflect')

        ###################################################################
        ####                                                  ####
        ###################################################################
        # "Compression"
        image = rgb_to_ycbcr_jpeg(image)
        if self.downsample_c:
            y, cb, cr = downsampling_420(image)
        else:
            # Splitting the image into Y, Cb, and Cr channels along the channel dimension
            y, cb, cr = torch.chunk(image, 3, dim=-1)
        components = {'y': y, 'cb': cb, 'cr': cr}
        ###################################################################
        ####                                                  ####
        ###################################################################
        for k in components.keys():
            comp = components[k]
            comp = image_to_patches(comp)
            comp = dct_8x8(comp)
            comp = c_quantize(comp, self.rounding,
                                factor) if k in ('cb', 'cr') else y_quantize(
                                    comp, self.rounding, factor)
            components[k] = comp    
        ###################################################################
        ####                                                  ####
        ###################################################################
        # "Decompression"
        for k in components.keys():
            comp = components[k]
            comp = c_dequantize(comp, factor) if k in ('cb', 'cr') else y_dequantize(
                comp, factor)
            comp = idct_8x8(comp)
            if k in ('cb', 'cr'):
                if self.downsample_c:
                    comp = patches_to_image(comp, int(height/2), int(width/2))
                else:
                    comp = patches_to_image(comp, height, width)
            else:
                comp = patches_to_image(comp, height, width)
            components[k] = comp
        ###################################################################
        ####                                                  ####
        ###################################################################
        # Assuming components is a dictionary with 'y', 'cb', 'cr' keys
        y, cb, cr = components['y'], components['cb'], components['cr']

        if self.downsample_c:
            image = upsampling_420(y, cb, cr)
        else:
            # Stack the Y, Cb, and Cr channels along the last dimension
            image = torch.stack((y, cb, cr), dim=-1)

        # Convert from YCbCr to RGB
        image = ycbcr_to_rgb_jpeg(image)

        # Crop to original size
        if orig_height != height or orig_width != width:
            #image = image[:, top:-bottom, left:-right]
            image = image[:, :-vpad, :-wpad]

        # Hack: RGB -> YUV -> RGB sometimes results in incorrect values
        # Clipping the image values to the range [0, 255]
        image = torch.clamp(image, min=0., max=255.)

        # Normalizing the image to the range [0, 1]
        image /= 255.

        return image
###################################################################
def diff_round(x):
    return torch.round(x) + (x - torch.round(x)) ** 3

###################################################################
####                                                  ####
###################################################################
def rgb_to_ycbcr_jpeg(image):
    # Define the conversion matrix and shift values
    matrix = torch.tensor(
        [[0.299, 0.587, 0.114], 
         [-0.168736, -0.331264, 0.5],
         [0.5, -0.418688, -0.081312]], 
        dtype=torch.float32
    ).T  # Transpose to match the TensorFlow implementation

    shift = torch.tensor([0., 128., 128.], dtype=torch.float32)

    # Perform matrix multiplication and add the shift
    result = torch.tensordot(image, matrix, dims=1) + shift

    # Ensure the output shape matches the input
    result = result.view_as(image)

    return result
###################################################################
####                                                  ####
###################################################################
def downsampling_420(image):
    # Input: batch x height x width x 3
    # Output: tuple of length 3
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2

    # Splitting the image into Y, Cb, and Cr channels
    y, cb, cr = torch.chunk(image, 3, dim=-1)

    # Apply average pooling for chroma subsampling (4:2:0)
    cb = F.avg_pool2d(cb.permute(0, 3, 1, 2), kernel_size=2, stride=2, padding=0)
    cr = F.avg_pool2d(cr.permute(0, 3, 1, 2), kernel_size=2, stride=2, padding=0)

    # Remove unnecessary dimensions
    y = y.squeeze(-1)
    cb = cb.squeeze(1)
    cr = cr.squeeze(1)

    return y, cb, cr


###################################################################
####                                                  ####
###################################################################
def image_to_patches(image):
    # Input: batch x height x width
    # Output: batch x (h*w/64) x 8 x 8

    k = 8
    batch_size, height, width = image.shape

    # Reshape the image into blocks of size 8x8
    image_reshaped = image.view(batch_size, height // k, k, width // k, k)
    
    # Transpose to bring the blocks together
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    
    # Reshape to get the final patches
    patches = image_transposed.reshape(batch_size, -1, k, k)

    return patches

###################################################################
####                                                  ####
###################################################################
def dct_8x8(image):
    # Subtract 128 to shift the range
    image = image - 128

    # Precompute the DCT transform tensor
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
    
    # Convert tensor to a PyTorch tensor
    tensor = torch.from_numpy(tensor).to(image.device)

    # Define the scaling factors (alpha)
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7, dtype=np.float32)
    scale = np.outer(alpha, alpha) * 0.25
    scale = torch.from_numpy(scale).to(image.device)

    # Perform the DCT using tensordot (axes=2 equivalent)
    result = scale * torch.tensordot(image, tensor, dims=2)

    # Ensure the output shape matches the input
    result = result.view_as(image)

    return result

###################################################################
####                                                  ####
###################################################################
# 5. Quantizaztion
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T
def c_quantize(image, rounding, factor=1):
    image = image / (c_table * factor)
    image = rounding(image)
    return image

###################################################################
####                                                  ####
###################################################################
# 5. Quantizaztion
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T
def y_quantize(image, rounding, factor=1):
    image = image / (y_table * factor)
    image = rounding(image)
    return image

###################################################################
####                                                  ####
###################################################################

def c_dequantize(image, factor=1):
    return image * (c_table * factor)

# -5. Dequantization
def y_dequantize(image, factor=1):
    return image * (y_table * factor)

###################################################################
####                                                  ####
###################################################################

# -4. Inverse DCT
def idct_8x8_ref(image):
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    alpha = np.outer(alpha, alpha)
    image = image * alpha

    result = np.zeros((8, 8), dtype=np.float32)
    for u, v in itertools.product(range(8), range(8)):
        value = 0
        for x, y in itertools.product(range(8), range(8)):
            value += image[x, y] * np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                (2 * v + 1) * y * np.pi / 16)
        result[u, v] = value
    return result * 0.25 + 128

def idct_8x8(image):
    # Define the scaling factors (alpha)
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7, dtype=np.float32)
    alpha = np.outer(alpha, alpha)
    alpha = torch.from_numpy(alpha).to(image.device)

    # Apply the scaling to the image
    image = image * alpha

    # Precompute the IDCT transform tensor
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos((2 * v + 1) * y * np.pi / 16)

    # Convert the tensor to a PyTorch tensor
    tensor = torch.from_numpy(tensor).to(image.device)

    # Perform the IDCT using tensordot (similar to TensorFlow)
    result = 0.25 * torch.tensordot(image, tensor, dims=2) + 128

    # Ensure the output shape matches the input
    result = result.view_as(image)

    return result

###################################################################
####                                                  ####
###################################################################
def patches_to_image(patches, height, width):
    # Input: batch x (h*w/64) x 8 x 8
    # Output: batch x h x w
    k = 8
    batch_size = patches.shape[0]

    # Reshape the patches to form the grid
    image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
    
    # Transpose to rearrange the patches correctly
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    
    # Reshape to form the final image
    image = image_transposed.reshape(batch_size, height, width)

    return image
###################################################################
####                                                  ####
###################################################################
def upsampling_420(y, cb, cr):
    # Input:
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2
    # Output:
    #   image: batch x height x width x 3

    def repeat(x, k=2):
        batch_size, height, width = x.shape
        x = x.unsqueeze(-1).unsqueeze(-1)  # Add dimensions for tiling
        x = x.repeat(1, 1, k, 1, k)        # Repeat along height and width
        x = x.permute(0, 1, 3, 2, 4).reshape(batch_size, height * k, width * k)
        return x

    cb = repeat(cb)
    cr = repeat(cr)

    # Stack the Y, Cb, and Cr channels along the last dimension
    image = torch.stack((y, cb, cr), dim=-1)

    return image

###################################################################
####                                                  ####
###################################################################
def ycbcr_to_rgb_jpeg(image):
    # Define the conversion matrix and shift values
    matrix = torch.tensor(
        [[1., 0., 1.402],
         [1, -0.344136, -0.714136],
         [1, 1.772, 0]],
        dtype=torch.float32
    ).T  # Transpose to match TensorFlow implementation

    shift = torch.tensor([0, -128, -128], dtype=torch.float32).to(image.device)

    # Perform matrix multiplication and add the shift
    result = torch.tensordot(image + shift, matrix, dims=1)

    # Ensure the output shape matches the input
    result = result.view_as(image)

    return result

###################################################################
####                                                  ####
###################################################################