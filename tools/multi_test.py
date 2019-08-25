# from mmdet.apis.multi_inference import init_detector, inference_detector, show_result
from mmdet.apis.multi_inference import init_detector, inference_detector, show_result
import os
import argparse
import sys
import torch

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
  config_file = "./configs/rscup/htc_sy.py"
  checkpoint_file = "./work_dirs/htc_sy/epoch_12.pth"
  model = init_detector(config_file, checkpoint_file)
  print(model.CLASSES)
  img_paths = ['./result/demo/7.jpg', './result/demo/pic2.jpg']
  result = inference_detector(model, img_paths)
  savename = "./result/demo/pic_det7.png"
  show_result(img_paths[0], result[0], class_names=[], score_thr=0.0, out_file=savename)
  savename = "./result/demo/pic_det8.png"
  show_result(img_paths[1], result[1], class_names=[], score_thr=0.0, out_file=savename)

if __name__ == '__main__':
    main()