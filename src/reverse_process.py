import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from sampler import sampling
from glob import glob
import numpy as np
import pandas as pd

from tqdm import tqdm
import nibabel as nib
import argparse


if __name__ == 'main':

    parser = argparse.ArgumentParser(description="Generate MRI synthetic sample")
    parser.add_argument("--age", type=int, required=True, help="Age of the subject")
    parser.add_argument("--sex", type=int, choices=[1, 2], required=True, help="Sex of the subject (1 for male, 2 for female)")
    args = parser.parse_args()

    # Use arguments
    age = args.age
    sex = args.sex

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    model_name = 'ddpm_3D_tissuemask_4_wm_gm_1_2'
    model = torch.load(f'/DATA1/leehw/camcan/models_for_brain_synthesis/{model_name}/unet_1000epochs.pt', map_location = device)

    condition = torch.Tensor(np.array([[int(age)], [int(sex)]]).reshape([1,1,2]).astype('float32')).to(device)
    sample = sampling((72, 96, 72), device, model, condition).detach().cpu().numpy()[0,0]

    array = (np.clip(sample, 0, 1)*255).astype('uint8')

    pixdim = [-1,  2,  2,  2,  1,  1,  1,  1]
    pixdim[1] = 2
    pixdim[2] = 2
    pixdim[3] = 2
    pixdim[5] = 2
    pixdim[6] = 2
    pixdim[7] = 2
    affine = [[  -2,    0,    0,   90],
             [   0,    2,    0, -126],
             [   0,    0,    2,  -72],
             [   0,    0,    0,    1]]

    units = 0
    nifti_image = nib.Nifti1Image(array, affine = affine)

    nifti_image.header['pixdim'] = pixdim
    nifti_image.header['xyzt_units'] = units


    if sex == 1:
        nib.save(nifti_image, f'./{age}years_male.nii.gz')
    elif sex == 2:
        nib.save(nifti_image, f'./{age}years_female.nii.gz')
        


