import nibabel as nib
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from monai import transforms
import torch

import yaml
from easydict import EasyDict

from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer



class CamCAN_2mmDataset_with_tissuemask_3D(Dataset):
    def __init__(self, train_val):
        self.train_val = train_val

        self.data_info = pd.read_csv('/DATA1/leehw/camcan/{}.csv'.format(self.train_val)).loc[:]


    def __len__(self):
        return self.data_info.shape[0]
    def __getitem__(self, idx):
        
        subject = self.data_info['Subject'][idx]
        condition = np.array([[int(self.data_info['Age'][idx])], [int(self.data_info['Sex'][idx])]]).reshape([1,2]).astype('float32')
        #condition = np.array([int(self.data_info['Age'][idx])]).reshape([1,1]).astype('float32')
        
        if self.train_val == 'train':  
            data = nib.load(read_config('config.yaml').data.params.train.params.data_path+'sub-{}_defaced_T1.nii.gz'.format(subject))
            tissuemask_csf = nib.load('/DATA1/leehw/camcan/tissuemask/sub-{}_defaced_T1_pve_0.nii.gz'.format(subject))
            tissuemask_csf = tissuemask_csf.get_fdata().astype(read_config('config.yaml').data.params.dtype)
            tissuemask_csf = transforms.EnsureChannelFirst(channel_dim='no_channel')(torch.from_numpy(tissuemask_csf))

            tissuemask_gm = nib.load('/DATA1/leehw/camcan/tissuemask/sub-{}_defaced_T1_pve_1.nii.gz'.format(subject))
            tissuemask_gm = tissuemask_gm.get_fdata().astype(read_config('config.yaml').data.params.dtype)
            tissuemask_gm = transforms.EnsureChannelFirst(channel_dim='no_channel')(torch.from_numpy(tissuemask_gm))

            tissuemask_wm = nib.load('/DATA1/leehw/camcan/tissuemask/sub-{}_defaced_T1_pve_2.nii.gz'.format(subject))
            tissuemask_wm = tissuemask_wm.get_fdata().astype(read_config('config.yaml').data.params.dtype)
            tissuemask_wm = transforms.EnsureChannelFirst(channel_dim='no_channel')(torch.from_numpy(tissuemask_wm))

            tissuemasks = [tissuemask_csf, tissuemask_gm, tissuemask_wm]
        elif self.train_val == 'val':
            data = nib.load(read_config('config.yaml').data.params.val.params.data_path+'sub-{}_defaced_T1.nii.gz'.format(subject))
        array = data.get_fdata()[9:81, 6:102, 7:79] # 91x109x91 데이터를 72x96x72로 crop
        array = (array/255.0).astype(read_config('config.yaml').data.params.dtype)
        array = transforms.EnsureChannelFirst(channel_dim='no_channel')(torch.from_numpy(array))
        #array = transforms.SpatialPad(spatial_size=[96, 128, 96])(array)
        #array = transforms.Resize(spatial_size=[72, 96, 72])(array)
        if self.train_val == 'train':
            return {'images':array, 'condition': condition, 'tissuemask': tissuemasks}
        elif self.train_val =='val':
            return {'images':array, 'condition': condition}
def read_config(conf):
    with open(conf, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    return config


class CamCAN_2mmDataset_with_tissuemask_2D(Dataset):
    def __init__(self, train_val):
        self.train_val = train_val

        self.data_info = pd.read_csv('/DATA1/leehw/camcan/{}.csv'.format(self.train_val)).loc[:]


    def __len__(self):
        return self.data_info.shape[0]
    def __getitem__(self, idx):
        
        subject = self.data_info['Subject'][idx]
        condition = np.array([[int(self.data_info['Age'][idx])], [int(self.data_info['Sex'][idx])]]).reshape([1,2]).astype('float32')
        #condition = np.array([int(self.data_info['Age'][idx])]).reshape([1,1]).astype('float32')
        
        if self.train_val == 'train':  
            data = nib.load(read_config('config.yaml').data.params.train.params.data_path+'sub-{}_defaced_T1.nii.gz'.format(subject))
            tissuemask = nib.load('/DATA1/leehw/camcan/tissuemask/sub-{}_defaced_T1_pve_2.nii.gz'.format(subject))
            tissuemask = tissuemask.get_fdata().astype(read_config('config.yaml').data.params.dtype)[:,:,36]
            tissuemask = transforms.EnsureChannelFirst(channel_dim='no_channel')(torch.from_numpy(tissuemask))
        elif self.train_val == 'val':
            data = nib.load(read_config('config.yaml').data.params.val.params.data_path+'sub-{}_defaced_T1.nii.gz'.format(subject))
        array = data.get_fdata()[9:81, 6:102, 7:79][:,:,36] # d drive의 cropped_72_96_72_normalization_4_9_81_6_102_7_79 데이터와 같음!
        array = (array/255.0).astype(read_config('config.yaml').data.params.dtype)
        array = transforms.EnsureChannelFirst(channel_dim='no_channel')(torch.from_numpy(array))
        #array = transforms.SpatialPad(spatial_size=[96, 128, 96])(array)
        #array = transforms.Resize(spatial_size=[72, 96, 72])(array)
        if self.train_val == 'train':
            return {'images':array, 'condition': condition, 'tissuemask': tissuemask}
        elif self.train_val =='val':
            return {'images':array, 'condition': condition}
def read_config(conf):
    with open(conf, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    return config

    


def sampling_3D(input_shape=(128, 128, 96), device=None, unet_model=None, condition=None):

    # scheduler setting
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
    scheduler.set_timesteps(num_inference_steps=1000)

    inferer = DiffusionInferer(scheduler)
  
    noise = torch.randn((1, 1, input_shape[0], input_shape[1], input_shape[2]))
    #noise = torch.randn((1, 3, 32, 36, 32))q
    noise = noise.to(device)
    unet_model.eval()
    if condition is not None:
        synthetic_images = inferer.sample(
        input_noise=noise, diffusion_model=unet_model, scheduler=scheduler, conditioning=condition
        )
    else:
        synthetic_images = inferer.sample(
            input_noise=noise, diffusion_model=unet_model, scheduler=scheduler
        )
    return synthetic_images

def sampling_2D(input_shape=(128, 128), device=None, unet_model=None, condition=None):

    # scheduler setting
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
    scheduler.set_timesteps(num_inference_steps=1000)

    inferer = DiffusionInferer(scheduler)
  
    noise = torch.randn((1, 1, input_shape[0], input_shape[1]))
    #noise = torch.randn((1, 3, 32, 36, 32))q
    noise = noise.to(device)
    unet_model.eval()
    if condition is not None:
        synthetic_images = inferer.sample(
        input_noise=noise, diffusion_model=unet_model, scheduler=scheduler, conditioning=condition
        )
    else:
        synthetic_images = inferer.sample(
            input_noise=noise, diffusion_model=unet_model, scheduler=scheduler
        )
    return synthetic_images


def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()



# if __name__ == "__main__":
#     train_ds = CamCAN_2mmDataset('train')
#     train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, )#, shuffle=True, num_workers=8, persistent_workers=True)
#     print(f'Image shape {train_ds[0]["images"].shape}')
