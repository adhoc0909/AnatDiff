import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.transforms import (
    EnsureChannelFirstd,
    CenterSpatialCropd,
    Compose,
    Lambdad,
    LoadImaged,
    Resized,
    ScaleIntensityd,
)
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from utils import CamCAN_2mmDataset_with_tissuemask_2D, read_config, sampling_2D
import pickle 


print_config()

set_determinism(42) # Set deterministic training for reproducibility



config = read_config('config.yaml')


train_ds = CamCAN_2mmDataset_with_tissuemask_2D('train')
val_ds = CamCAN_2mmDataset_with_tissuemask_2D('val')
train_loader = DataLoader(train_ds, batch_size=config.data.params.batch_size, shuffle=True, num_workers=8, )#, shuffle=True, num_workers=8, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=config.data.params.batch_size, shuffle=True, num_workers=8, )
print(f'Image shape {train_ds[0]["images"][0].shape}')
#####################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
#####################################################################################################



affine = nib.load(config.data.params.train.params.data_path+'sub-CC110033_defaced_T1.nii.gz').affine # 모든 데이터의 affine을 sub-CC110037과 똑같이 적용하여 sampling


device = torch.device("cuda")

model = DiffusionModelUNet(
    spatial_dims=config.model.params.spatial_dims,
    in_channels=config.model.params.in_channels,
    out_channels=config.model.params.out_channels,
    num_channels=config.model.params.num_channels,
    attention_levels=config.model.params.attention_levels,
    num_head_channels=config.model.params.num_head_channels,
    num_res_blocks=config.model.params.num_res_blocks,
    with_conditioning=True,
    cross_attention_dim=2
)

#model = torch.load('/DATA1/leehw/camcan/models_for_brain_synthesis/ddpm_tissuemask/unet_100epochs.pt', map_location = device)

model.to(device)


scheduler = DDPMScheduler(num_train_timesteps=config.model.params.num_train_timesteps, 
                          schedule=config.model.params.schedule, 
                          beta_start=config.model.params.beta_start, 
                          beta_end=config.model.params.beta_end)


inferer = DiffusionInferer(scheduler)

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.model.base_learning_rate)

########################### training ############################
n_epochs = config.model.params.n_epochs
val_interval = config.model.params.val_interval

epoch_loss_list = []
val_epoch_loss_list = []

scaler = GradScaler()
total_start = time.time()
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["images"].to(device)
        conditions = batch['condition'].to(device)
        tissuemask = batch['tissuemask'].to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(images).to(device)

            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()
            anatomy_weight = 0.5
            # if timesteps < 500:
            #     noise = noise + (anatomy_weight * tissuemask)

            # Get model prediction
            
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps, condition=conditions)

            # loss = F.mse_loss(noise_pred.float(), (noise+(tissuemask*anatomy_weight)).float())
            loss = F.mse_loss(noise_pred.float(), noise.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_loss_list.append(epoch_loss / (step + 1))

    # if (epoch + 1) % val_interval == 0:
    #     model.eval()
    #     val_epoch_loss = 0
    #     for step, batch in enumerate(val_loader):
    #         images = batch["images"].to(device)
    #         conditions = batch['condition'].to(device)
    #         noise = torch.randn_like(images).to(device)
    #         with torch.no_grad():
    #             with autocast(enabled=True):
    #                 timesteps = torch.randint(
    #                     0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
    #                 ).long()

    #                 # Get model prediction
    #                 noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps, condition=conditions)
    #                 val_loss = F.mse_loss(noise_pred.float(), noise.float())

    #         val_epoch_loss += val_loss.item()
    #         progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
    #     val_epoch_loss_list.append(val_epoch_loss / (step + 1))

    #     # Sampling image during training
    #     image = torch.randn((1, 1, 32, 40, 32))
    #     image = image.to(device)
    #     scheduler.set_timesteps(num_inference_steps=1000)
    #     with autocast(enabled=True):
    #         image = inferer.sample(input_noise=image, diffusion_model=model, scheduler=scheduler)


    if epoch % 100 == 0:
        torch.save(model, '/DATA1/leehw/camcan/models_for_brain_synthesis/ddpm_2D/unet_{}epochs.pt'.format(epoch))

    if epoch % 100 ==0:
        random_age = np.random.randint(20, 81)
        random_sex = np.random.randint(0, 2)
        condition = torch.Tensor(np.array([[random_age], [random_sex]]).reshape([1,1,2]).astype('float32')).to(device)
        sample = sampling_2D((72, 96), device, model, condition).detach().cpu().numpy()[0][0]
        nifti = nib.Nifti1Image(sample, affine=affine)
        nifti.to_filename('/DATA1/leehw/camcan/models_for_brain_synthesis/ddpm_2D/{}epochs_result.nii.gz'.format(epoch))
        
        with open('/DATA1/leehw/camcan/models_for_brain_synthesis/ddpm_2D/unet_history.pkl', 'wb') as f:
            pickle.dump(epoch_loss_list, f)



with open('/DATA1/leehw/camcan/models_for_brain_synthesis/ddpm_2D/unet_history.pkl', 'wb') as f:
    pickle.dump(epoch_loss_list, f)


torch.save(model, '/DATA1/leehw/camcan/models_for_brain_synthesis/ddpm_2D/unet_1000epochs.pt')

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")