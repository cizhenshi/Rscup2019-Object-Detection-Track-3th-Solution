# from mmdet.apis.multi_inference import init_detector, inference_detector, show_result
from mmdet.apis.inference import init_detector, inference_detector, show_result
import os
import argparse
import sys
import torch
from tqdm import tqdm as tqdm
import torch.multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser(description='in and out imgs')
    parser.add_argument('--config', dest='config',help='config_file',default=None, type=str)
    parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default=None, type=str)

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def main():
  # args = parse_args()
  # config_file = args.config
  # checkpoint_file = args.checkpoint
  # p = Pool(4)
  config_file = "./configs/rscup/htc_res50.py"
  checkpoint_file = "./work_dirs/htc_res50/epoch_12.pth"
  model = init_detector(config_file, checkpoint_file)
  print(model.CLASSES)
  img = './result/demo/7.jpg'
  results = inference_detector(model, img)

if __name__ == '__main__':
    main()