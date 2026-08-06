


"""
@--01.04.2026--@
Author: github/farhadsh1992
INFO:
LAST_UPDATE:
"""



import time
import torch
import torch.multiprocessing as mp

from DocSafe import Trainer
from DocSafe.args import train_setting, noise_setting
from DocSafe.augmentors import Augmentor, Warper
from DocSafe.monitors import tensorboard_monitor#, wandb_monitors
from Tools_nvidia_torch.torch_utils import Configure_GPU


if __name__ == "__main__":

    args = train_setting()
    args_noise = noise_setting()

    docsafe_trainer = Trainer(
                    model="M1",
                    args = args,
                    args_noise = args_noise,
                    )
    # NOTE: Trainer currently has no load_network() dispatcher (unlike encoder()/decoder()
    # in encoder_router.py/decoder_router.py, which expose one). It only has
    # Load_M1_networks()/Load_M2_networks()/Load_M3_networks(); call the one matching
    # `model=` above directly, e.g.:
    #   docsafe_trainer.Load_M1_networks(device="cpu")

    docsafe_trainer.load_data(
                        data_path="./data/",
                        mask_path="./data/masks/",
                       )
    docsafe_trainer.load_optimizers(optimizer_type="Adam")
    docsafe_trainer.load_loss_functions()


    live_monitor = tensorboard_monitor(
                    myargs           = args,
                    args_noise       = args_noise,
                    log_dir          = args.save_dir + "logs/",
                    display_interval = args.display_interval,
                    gpu_id     = args.rank, # which GPU this process is running on.
                    world_size = args.world_size, # number of GPUs you want to use
                    devices    = docsafe_trainer.devices[0]
                )
    docsafe_trainer.upload_live_monitor(Live_Monitors = live_monitor)



    augmentor = Augmentor(
                    args       = args_noise,
                    batch_size = args.batch_size,
                    image_size = args.image_size,
                    device     = docsafe_trainer.devices[0] )


    warper  = Warper(
                    args       = args,
                    noise_args = args_noise,
                    batch_size = args.batch_size,
                    image_size = args.image_size,
                    device     = docsafe_trainer.devices[0])

    docsafe_trainer.upload_augmentation(augmentor     = augmentor,
                                        warper_router  = warper)



    start_time = time.time()
    devices = Configure_GPU(args)
    world_size = torch.cuda.device_count()

    print("world_size", world_size)
    # NOTE: Trainer.train(rank, world_size) itself is still unfinished — its body
    # references undefined names (`args` instead of `self.args`, `ddp_setup`, `max_epochs`)
    # ported over from train_steps.py's main_test(). It will raise NameError until that's
    # fixed. Once fixed, calling convention should look like:
    if world_size <= 1:
        docsafe_trainer.train(rank=0, world_size=world_size)
    else:
        mp.spawn(docsafe_trainer.train,
                args=(world_size,),
                nprocs=world_size)