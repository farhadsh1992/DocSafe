


"""
@--03.02.2025--@
Author: github/farhadsh1992
INFO:
     
LAST_UPDATE:
"""
################################################
# from configs import data_configs
# from datasets.dataset_fetcher import DatasetFetcher
################################################
from FarhadCV.Tools import tcolors, bcolors, estimator, read_files
import numpy as np
from PIL import Image
import cv2
################################################
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from matplotlib import cm
import os

from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from torch.utils.data.distributed import DistributedSampler
################################################
# import .transforms_config as 
################################################

class Dataset_Router():
    """
    ---------------------------------------
    INFO:
    """
    def __init__(self, args, 
                 TRAIN_COVER:str,
                 TEST_COVER:str,
                 rank, 
                 world_size,
                 device:str):
        
        ################################
        ###                           ###
        ################################

        self.args = args
        self.device = device
        self.TRAIN_COVER = TRAIN_COVER
        self.TEST_COVER = TEST_COVER
        self.rank = rank
        self.world_size = world_size
        self.batch_size = self.args.batch_size
        self.message_shape2 = args.mess_reshape2
        ################################
        ###                           ###
        ################################

    # @staticmethod
    def load(self)->(Dataset, Dataset):
        ## Check the data configs:
        # if self.args.dataset_type not in data_configs.DATASETS.keys():
        #     raise Exception('{} is not a valid dataset_type'.format(self.args.dataset_type))
        # print()
        # print(tcolors.BLUE+bcolors.WHITE, 'Loading dataset for {}'.format(self.args.dataset_type), tcolors.ENDC)

        transform_cover = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        

        transform_message_en = transforms.Compose([
                transforms.Resize((16, 16)),
                # transforms.Resize((self.message_shape2[0], self.message_shape2[1])),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation(degrees=(-5,5)),
                transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                
                ])
        transform_message_de = transforms.Compose([
                transforms.Resize((16, 16)),
                # transforms.Resize((self.message_shape2[0], self.message_shape2[1])),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                
                ])
        
        transform_boxes = transforms.Compose([
                # transforms.Resize((16, 16)),
                transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])


        train_dataset_router = ImagesDataset(
                                         args = self.args,
                                         cover_root = self.TRAIN_COVER,
                                         cover_transform = transform_cover,
                                         message_transform_en = transform_message_en,
                                         message_transform_de = transform_message_de,

                                         transform_boxes = transform_boxes,
                                         secret_size = self.args.secret_size, 
                                         pad_size = self.args.pad_size, 
                                         image_size_qr_in = self.args.message_shape[0], 
                                         batch_size = self.args.batch_size,
                                         device = self.device)
        test_dataset_router = ImagesDataset(
                                        args = self.args,
                                        cover_root = self.TEST_COVER ,
                                        cover_transform = transform_cover,
                                        message_transform_en = transform_message_en,
                                        message_transform_de = transform_message_de,
                                        transform_boxes = transform_boxes,
                                        secret_size = self.args.secret_size, 
                                        pad_size = self.args.pad_size, 
                                        image_size_qr_in = self.args.message_shape[0], 
                                        batch_size = self.args.batch_test,
                                        device = self.device)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset_router,
                    num_replicas = self.world_size,
                    rank         = self.rank,
                    shuffle      = True,
                )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_dataset_router,
                    num_replicas = self.world_size,
                    rank         = self.rank,
                    shuffle      = True,
                )
        # per_gpu_batch_size = self.batch_size // self.world_size
        

        train_dataset_router = prepare_dataloader(train_router  = train_dataset_router, 
                                                  train_sampler = train_sampler, 
                                                  batch_size    = 1,#self.args.batch_size,
                                                  num_workers   = 0#self.world_size
                                                  )
        test_dataset_router = prepare_dataloader(train_router  = test_dataset_router, 
                                                 train_sampler = test_sampler,
                                                 batch_size    = 1,#self.args.batch_test,
                                                 num_workers   = 0#self.world_size
                                                 )
        return train_dataset_router, test_dataset_router


class ImagesDataset(Dataset):
    """
    INFO:
        - 
    """

    def __init__(self, 
                args, 
                cover_root:str=None, 
                cover_transform=None, 
                message_transform_en=None, 
                message_transform_de=None, 
                transform_boxes = None, 
                secret_size:int=256, 
                pad_size:int=None, 
                image_size_qr_in:int=16,
                batch_size:int=2, 
                device:str=None):
        super(ImagesDataset, self).__init__()

        self.iepoch = 0
        self.device = device
        self.args   = args
        self.batch_size  = batch_size
        self.cover_root   = cover_root 
        self.image_size = args.image_size
        self.steps_per_epoch  = args.steps_per_epoch

        self.TRAIN_MONODepth = args.monocular_depth_path
        # self.target_paths = target_root
        self.cover_images1 = read_files(cover_root)
        self.cover_images2 = read_files(self.TRAIN_MONODepth)
        # self.target_images = read_files(target_root)
        self.cover_images = compare_lists(self.cover_images1, self.cover_images2)

        self.cover_transform = cover_transform
        self.message_transform_en = message_transform_en
        self.message_transform_de = message_transform_de

        self.transform_boxes = transform_boxes
        # self.target_transform = target_transform

        self.secret_size = secret_size
        self.pad_size = pad_size
        self.image_size_qr_in = image_size_qr_in

        threshold = [0.5] #0.4
        Value2    = [0]
        mask = Image.open(args.pattern_img).convert('RGB').resize((self.image_size, self.image_size))
        mask = self.cover_transform(mask)
        mask = torch.where(mask < threshold[0], Value2[0], mask)
        
        self.mask_batch = [mask for i in range(self.batch_size)]
        self.mask_batch = torch.stack(self.mask_batch)
        # edges = generate_circle_edges((256, 256), num_circles=15)

        # self.mask_batch = [edges for i in range(self.batch_size)]
        # self.mask_batch = torch.stack(self.mask_batch)

    def __len__(self):
        return self.steps_per_epoch#len(self.cover_images)
    def __getitem__(self, index:int):

        
       
        cover_img = np.random.choice(self.cover_images, self.batch_size)
        # prob_mask = np.random.choice([0, 1], p=[0.70, 0.30])
        
        prob_mask = np.random.choice([0, 1], p=[0.70, 0.30])
        from_cover = self.cover_root  + cover_img[i]
        pathMD = self.TRAIN_MONODepth + cover_img[i]

        #######################################################################
        #####    READ IMAGE           #####
        #######################################################################
        cover_image = Image.open(from_cover).convert('RGB').resize((self.image_size, self.image_size))
        cover_image = self.cover_transform(cover_image)


        #######################################################################
        #####    READ Dpeth  monocular-depth-estimation         #####
        #######################################################################
        if prob_mask == 0: 
            # threshold = np.random.choice([.3,.3,.4,.34, 0.3,0.2, 0.1], 1)
            # Value2 = np.random.choice([0.2,.5,.3,.3, 0.4,0.4, 0.2,0.2,0.2,0.2], 1)
            threshold = [0.4]
            Value2    = [0]

            depthint = Image.open(pathMD).convert('RGB').resize((self.image_size, self.image_size))
            depthint = self.cover_transform(depthint)
            depthint = torch.where(depthint < threshold[0], Value2[0], depthint)
            # depthint = torch.where(depthint > threshold[0], 1.0, depthint)
        elif prob_mask == 1: 
            depthint = torch.ones(cover_image.shape)
        # else:
        #     depthint =  generate_random_mask(shape=(3, 256, 256), min_square_size=64)
        #######################################################################
        #####    READ Message           #####
        #######################################################################
        message_in_en, message_in_de = self.produce_random_massage()
        message_in_en = self.message_transform_en(message_in_en)
        message_in_de = self.message_transform_de(message_in_de)


        #######################################################################
        #####    READ Coordinates        #####
        #######################################################################
        ## box > tuple:(y_min, y_max, x_min, x_max)
        # Remove the file extension
        # name_without_extension = os.path.splitext(cover_img[i])[0]
        # box = read_coordinates(dataset_path = self.TRAIN_BOX, 
        #                          image_name = name_without_extension)
        # #
        # ####--------------------------------------------------------------------------
        # # print(tcolors.RED,"box: ",box, "|", cover_img[i] ,tcolors.ENDC)
        # if box == None:
        #     os.remove(self.cover_root + cover_img[i])
        # y_min,y_max, x_min, x_max = box
        # y_min = min(300, max(0, y_min))
        # y_max = min(300, max(0, y_max))
        # x_min = min(300, max(0, x_min))
        # x_max = min(300, max(0, x_max))
        # # box = box/256
        # # 
        # # box = [float(y_min)/256.0, float(y_max)/256.0, float(x_min)/256.0, float(x_max)/256.0]
        
        # ## detr form: [ymin, xmin, ymax, xmax]
        # box = [float(y_min)/300.0, float(x_min)/300.0, 
        #        float(y_max)/300.0, float(x_max)/300.0]
        # # box = np.array([y_min, y_max,x_min, x_max])
        # # box = self.transform_boxes(box)
        #######################################################################
        #####    READ IMAGE           #####
        #######################################################################
        image_batch.append(cover_image)
        code_Normal_en.append(message_in_en)
        code_Normal_de.append(message_in_de)

        # code_Normal_de.append(message_in_de)
        # list_boxes.append(box)
        list_dapth.append(depthint)


        image_batch = torch.stack(image_batch)
        code_Normal_en = torch.stack(code_Normal_en)
        code_Normal_de = torch.stack(code_Normal_de)

        list_dapth  = torch.stack(list_dapth)
        list_boxes  = torch.from_numpy(np.array(list_boxes))
        new_image   = image_batch * self.mask_batch
        self.iepoch = self.iepoch + 1
        # self.iepoch = torch.from_numpy(np.array([self.iepoch]))
        self.iepoch = torch.tensor([self.iepoch], dtype=torch.int)
        return (self.iepoch, image_batch, (code_Normal_en, code_Normal_de), image_batch, list_dapth, new_image)



    def produce_random_massage(self):
        ## generate a binery code randomly
        a_vector_message = (np.random.randint(2, size=(self.secret_size*1))).astype("uint8")
        a_vector_message = (a_vector_message).astype('float32')

        ## covert 1D binery code to 2D binery code - (shape=(reshape_size_1, reshape_size_2))
        a_vector_message2D = np.reshape(a_vector_message*255, (self.args.mess_reshape[0], 
                                                                self.args.mess_reshape[1], 1))
        # Add an extra row of ones (255) at the bottom to make it (14,14,1)
        ones_row = np.zeros((1, self.args.mess_reshape[1], 1), dtype='float32') * 255
        a_vector_message2D = np.vstack((a_vector_message2D, ones_row))

        # covert 2D binery code (gray) to 3D binery code (RGB) - (shape=(reshape_size_1, reshape_size_2, 3))
        a_vector_message2D = cv2.cvtColor(a_vector_message2D, cv2.COLOR_GRAY2RGB)

        ## add paddings
        a_vector_message2D_pad = cv2.copyMakeBorder(a_vector_message2D, 
                                top = self.pad_size, bottom = self.pad_size, 
                                left= self.pad_size, right = self.pad_size, 
                                borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
                                
        ## Become sure the size
        a_vector_message2D = cv2.resize(a_vector_message2D, 
                                        (10, 10), 
                                        interpolation=cv2.INTER_NEAREST)
        a_vector_message2D_pad = cv2.resize(a_vector_message2D_pad, 
                                        (16, 16), 
                                        interpolation=cv2.INTER_NEAREST)
        # a_vector_message2D_gray = cv2.cvtColor(a_vector_message2D, cv2.COLOR_RGB2GRAY)
        a_vector_message2D_pad_gray = cv2.cvtColor(a_vector_message2D_pad, cv2.COLOR_RGB2GRAY)
        a_vector_message2D_gray = cv2.cvtColor(a_vector_message2D, cv2.COLOR_RGB2GRAY)


        ## Transpose from HWC (normal format) to CHW (torch-format)
        # a_vector_message2D_en = a_vector_message2D.transpose(2,0,1)
        # a_vector_message2D_en = a_vector_message2D.transpose(2,1,0)
        # print(tcolors.RED,"a_vector_message2D_en:", a_vector_message2D_en.shape,tcolors.ENDC)
        # a_vector_message2D_en = Image.fromarray(np.uint8(a_vector_message2D_en))
        a_vector_message2D_en = Image.fromarray(np.uint8(a_vector_message2D_pad))
        a_vector_message2D_de = Image.fromarray(np.uint8(a_vector_message2D_pad_gray))


        ## Normalize the image
        
        # a_vector_message2D_en = (a_vector_message2D_en/255.0).astype('float32')

        return a_vector_message2D_en, a_vector_message2D_de
    



def prepare_dataloader(train_router: Dataset, 
                       train_sampler, 
                       batch_size: int,
                       num_workers):
    """
    If num_workers=0, each time the model needs a batch, the main thread will load data one-by-one. 
    It is simple, but slow if loading takes time 
    (especially if data is on disk or needs preprocessing).

    If num_workers=2, there are two parallel workers loading batches in the background.
    While the model is training on batch 1, workers can already be preparing batch 2 and batch 3.
    ➔ Faster and smoother training, especially for heavy datasets (like images, videos).
    
    """
    return DataLoader(
        train_router,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        # num_workers=num_workers,
        sampler=train_sampler,
    )
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


########################################################################
####                                                 ####    
########################################################################
def read_coordinates(dataset_path, image_name):
    """
    Read (y_min, y_max, x_min, x_max) from the corresponding text file.

    Args:
    - dataset_path (str): Path to the folder containing text files.
    - image_name (str): Name of the image file (e.g., 'image.jpg').

    Returns:
    - tuple: (y_min, y_max, x_min, x_max) as integers.
    """
    base_name = os.path.splitext(image_name)[0]
    text_file_path = os.path.join(dataset_path, f"{base_name}.txt")
    
    if os.path.exists(text_file_path):
        with open(text_file_path, 'r') as file:
            coords = file.read().strip()
            return tuple(map(int, coords.split(',')))
    else:
        print(f"Coordinates file for {image_name} does not exist.")
        return None
    

########################################################################
####                                                 ####    
########################################################################
def compare_lists(list1, list2):
    """
    Compares two lists and returns whether they are the same
    and a list of their common elements.

    Args:
        list1 (list): First list
        list2 (list): Second list

    Returns:
        tuple: (boolean, list) - True if lists are the same, False otherwise, 
               and a list of common elements.
    """
    is_same = list1 == list2  # Check if both lists are identical
    common_elements = list(set(list1) & set(list2))  # Find common elements
    
    return common_elements



def generate_circle_edges(size, num_circles=10, edge_thickness=1.5):
    """
    Generate binary black-white circle edge pattern as 2D numpy array.
    Black represents edges.
    """
    h, w = size
    y, x = np.ogrid[:h, :w]
    center = (h / 2, w / 2)
    radius = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    circle_map = np.sin(radius * num_circles * np.pi / min(h, w))  # sine wave pattern

    # Normalize and detect edges
    circle_map = gaussian_filter(circle_map, sigma=1)
    edge_map = sobel(circle_map)
    
    # Threshold to get binary edge
    edge_binary = (edge_map > edge_thickness * edge_map.mean()).astype(np.float32)

    return edge_binary


def generate_random_mask(shape=(3, 256, 256), min_square_size=32):
    """
    Generate a binary mask with shape (3, H, W), where all values are 0 except for one random square region set to 1.

    Args:
        shape (tuple): Shape of the mask (default is (3, 256, 256)).
        min_square_size (int): Minimum side length of the square.

    Returns:
        np.ndarray: The generated mask.
    """
    mask = np.zeros(shape, dtype=np.float32)
    
    _, H, W = shape
    
    # Random square size between min_square_size and half of the image size
    square_size = np.random.randint(min_square_size, min(H, W) // 2 + 1)
    
    # Random top-left corner
    top = np.random.randint(0, H - square_size + 1)
    left = np.random.randint(0, W - square_size + 1)
    
    # Set the square region to 1
    mask[:, top:top+square_size, left:left+square_size] = 1.0
    
    return torch.from_numpy(mask)