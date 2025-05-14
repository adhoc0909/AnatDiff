# import yaml
# from easydict import EasyDict

# with open('config.yaml', 'r') as f:
#     config = yaml.safe_load(f)
# config = EasyDict(config)

# print(config.data.params.train.params.data_path)

from utils import read_config
config = read_config('config.yaml')


print(config.data.params.train.params.data_path)
