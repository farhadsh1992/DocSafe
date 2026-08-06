

"""
@--03.02.2025--@
Author: github/farhadsh1992
INFO:
    -ref: 
      
    
LAST_UPDATE:
"""
import os
import sys
from pprint import pprint
from FarhadCV.Tools import tcolors, bcolors
import torch



###########################################################
####                                             ####
###########################################################
def CHECK_PYTHON_SETTING():
    import pkg_resources
    print(tcolors.RED)
    pprint({
        'PATH': os.environ['PATH'].split(os.pathsep),
        'PYTHONPATH': get_pythonpath(),
        'sys.path': sys.path,
        'sys.executable': sys.executable,
        'sys.prefix': sys.prefix,
        'sys.version_info': sys.version_info,
        'pkg_resources.working_set': list(pkg_resources.working_set),
    })
    print(tcolors.ENDC)

#####################################
def get_pythonpath():
    try:
        return os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        return None
###########################################################
####                                             ####
###########################################################
    

def Configure_GPU(args):

    ## Check that GPU is avaible 
    if torch.cuda.is_available():
        print(bcolors.WHITE+tcolors.BLUE,
              f"GPU is available: {torch.cuda.get_device_name(0)}"
              ,  tcolors.ENDC)
    else:
        print(tcolors.RED, "GPU is not available.",  tcolors.ENDC)

    devices_list = []
    numgpy = args.gpu_devices.split(",")
    for i in range(len(numgpy)):
        device = torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu")
        devices_list.append(device)
    devices_list.append("cpu")
    
   
    
    return devices_list



###########################################################
####                                             ####
###########################################################
    
import shutil

def check_free_space(drive_path):
    try:
        # Get the disk usage statistics for the given path
        disk_usage = shutil.disk_usage(drive_path)
        
        # Convert the values from bytes to GB
        total_space_gb = disk_usage.total / (1024 ** 3)
        used_space_gb = disk_usage.used / (1024 ** 3)
        free_space_gb = disk_usage.free / (1024 ** 3)
        
        return {
            "Path": drive_path,
            "Total Space (GB)": total_space_gb,
            "Used Space (GB)": used_space_gb,
            "Free Space (GB)": free_space_gb
        }
    except FileNotFoundError:
        return f"Error: The path '{drive_path}' does not exist."