import matplotlib.pyplot as plt
from utils import CamCAN_2mmDataset_with_tissuemask_3D
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.cuda.amp import autocast

from generative.inferers import DiffusionInferer
from generative.networks.schedulers import DDPMScheduler
from generative.networks.nets import DiffusionModelUNet
from monai import transforms
from monai.utils import first, set_determinism

from torch.utils.data import Dataset, DataLoader
from glob import glob
import numpy as np
import pandas as pd

from tqdm import tqdm
import nibabel as nib



def sampling(input_shape=(128, 128, 96), device=None, unet_model=None, condition=None):

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

model_name = 'ddpm_3D_tissuemask_4_csf_1_2'
model = torch.load(f'/DATA1/leehw/camcan/models_for_brain_synthesis/{model_name}/unet_1000epochs.pt', map_location = device)

data = nib.load('/DATA1/leehw/camcan/cropped_72_96_72_normalization_9_81_6_102_7_79/sub-CC110033_defaced_T1.nii.gz')
pixdim = data.header['pixdim']
pixdim[1] = 2
pixdim[2] = 2
pixdim[3] = 2
pixdim[5] = 2
pixdim[6] = 2
pixdim[7] = 2
affine = data.affine
units = data.header['xyzt_units']

# for i in range(10, 1000):
#     age = np.random.randint(18, 89)
#     sex = np.random.randint(0, 2)
#     # age = -5
#     # sex = 3
#     condition = torch.Tensor(np.array([[age], [sex]]).reshape([1,1,2]).astype('float32')).to(device)
#     sample = sampling((72, 96, 72), device, model, condition).detach().cpu().numpy()[0,0]
#     #sample = sampling((16, 16, 16), device, model, condition).detach().cpu().numpy()[0,0]



#     array = (np.clip(sample, 0, 1)*255).astype('uint8')

#     nifti_image = nib.Nifti1Image(array, affine = affine)

#     nifti_image.header['pixdim'] = pixdim
#     nifti_image.header['xyzt_units'] = units

#     if sex == 0:
#         sex = 'male'
#     else:
#         sex = 'female'
#     nib.save(nifti_image, '/home/leehw/project/brain_synthesis/data/synthetic/ddpm_3D_tissuemask_4_csf_1_2/{}_{}_{}.nii.gz'.format(i,age,sex))

phenotype = pd.read_csv('./phenotype.csv')

cnt = 322
#cnt = 0

#for age, sex in zip(phenotype[:322]['Age'], phenotype[:322]['Sex']):
for age, sex in zip(phenotype[322:]['Age'], phenotype[322:]['Sex']):    
#for age, sex in zip(phenotype[454:]['Age'], phenotype[454:]['Sex']):    
    # age = -5
    # sex = 3
    #if i%6 !=0:
    print(cnt)
    if sex == 1:
        
        condition = torch.Tensor(np.array([[int(age)], [int(sex)]]).reshape([1,1,2]).astype('float32')).to(device)
        sample = sampling((72, 96, 72), device, model, condition).detach().cpu().numpy()[0,0]
        #sample = sampling((16, 16, 16), device, model, condition).detach().cpu().numpy()[0,0]



        array = (np.clip(sample, 0, 1)*255).astype('uint8')

        nifti_image = nib.Nifti1Image(array, affine = affine)

        nifti_image.header['pixdim'] = pixdim
        nifti_image.header['xyzt_units'] = units

    
        sex = 'male'
        nib.save(nifti_image, f'/DATA1/leehw/camcan/models_for_brain_synthesis/synthetic/male/{model_name}/{cnt}_{age}_{sex}.nii.gz')
        
    cnt+=1
     


