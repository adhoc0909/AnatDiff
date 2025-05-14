from utils import CamCAN_2mmDataset_with_tissuemask_2D, CamCAN_2mmDataset_with_tissuemask_3D
from monai.data import DataLoader

train_ds = CamCAN_2mmDataset_with_tissuemask_3D('train')
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, )#, shuffle=True, num_workers=8, persistent_workers=True)
# print(f'Image shape {train_ds[0]["tissuemask"].min()}')

import pandas as pd
print(pd.Series((train_ds[0]['tissuemask']+1).reshape(-1)).value_counts())
# import pandas as pd
# import numpy as np

# df = pd.DataFrame(train_ds[0]["tissuemask"][0])
# unique_values = {}
# for col in df.columns:
#     unique_values[col] = df[col].unique()

# # 고유한 값 출력
# for col, values in unique_values.items():
#     print(f"Column {col}: {values}")