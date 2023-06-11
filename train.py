import argparse
import numpy as np
import os
import json 

import torch
import torch.multiprocessing as mp

from core.philly import ompi_size, ompi_local_size, ompi_rank, ompi_local_rank
from core.philly import get_master_ip, gpu_indices, ompi_universe_size
from core.utils import set_seed
from core.trainer import Trainer


parser = argparse.ArgumentParser(description="Pconv")
parser.add_argument('-c', '--config', type=str, default=None, required=True)
parser.add_argument('-n', '--name', default='base', type=str)
parser.add_argument('-m', '--mask', default=None, type=str)
parser.add_argument('-s', '--size', default=None, type=int)
parser.add_argument('-p', '--port', default='23455', type=str)
parser.add_argument('-e', '--exam', action='store_true')
args = parser.parse_args()

# 0 1 config
def main_worker(gpu, ngpus_per_node, config):
  if 'local_rank' not in config:
    config['local_rank'] = config['global_rank'] = gpu # local_rank = 0
  if config['distributed']: # false 不会进行分布式训练
    torch.cuda.set_device(int(config['local_rank']))
    print('using GPU {} for training'.format(int(config['local_rank'])))
    torch.distributed.init_process_group(backend = 'nccl', 
      init_method = config['init_method'],
      world_size = config['world_size'], 
      rank = config['global_rank'],
      group_name='mtorch'
    )
  set_seed(config['seed']) # 2020
  # 拼接得到 save_dir = '{save_dir}/pennet_dtd_square256'
  config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}'.format(config['model_name'], 
                                                                            config['data_loader']['name'],
                                                                            config['data_loader']['mask'],
                                                                            config['data_loader']['w']))
  if (not config['distributed']) or config['global_rank'] == 0:
    os.makedirs(config['save_dir'], exist_ok=True) # 创建文件夹  save_dir = '{save_dir}/pennet_dtd_square256'
    print('[**] create folder {}'.format(config['save_dir']))

  trainer = Trainer(config, debug=args.exam)
  trainer.train()

# Downloading: "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth" to /home/ywh/.cache/torch/hub/checkpoints/inception_v3_google-1a9a5a14.pth

if __name__ == "__main__":
  print('check if the gpu resource is well arranged on philly')  # 检查gpu资源是否安排良好
  assert ompi_size() == ompi_local_size() * ompi_universe_size()
  
  # 加载并读取配置文件
  config = json.load(open(args.config)) # 加载配置  args.config 是配置文件
  if args.mask is not None:
    config['data_loader']['mask'] = args.mask # 设置加载器的掩码 mask
  if args.size is not None:  # 一般 256
    config['data_loader']['w'] = config['data_loader']['h'] = args.size
  config['model_name'] = args.name # 模型名 pennet
  config['config'] = args.config # 配置文件路径

  # setup distributed parallel training environments
  world_size = ompi_size()
  ngpus_per_node = torch.cuda.device_count()
  if world_size > 1:
    config['world_size'] = world_size
    config['init_method'] = 'tcp://' + get_master_ip() + args.port
    config['distributed'] = True
    config['local_rank'] = ompi_local_rank()
    config['global_rank'] = ompi_rank()
    main_worker(0, 1, config)
  elif ngpus_per_node > 1:
    config['world_size'] = ngpus_per_node
    config['init_method'] = 'tcp://127.0.0.1:'+ args.port 
    config['distributed'] = True
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
  else: # 非分布式训练
    config['world_size'] = 1 
    config['distributed'] = False
    main_worker(0, 1, config)
