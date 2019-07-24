# from mmdet.apis.multi_inference import init_detector, inference_detector, show_result
from mmdet.apis.inference import init_detector, inference_detector, show_result
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
  args = parse_args()
  config_file = args.config
  checkpoint_file = args.checkpoint
  model = init_detector(config_file, checkpoint_file)
  print(model.CLASSES)
  root ='/mmdetection/work_dirs/aug/'
  savedir = '/result'
  img = './result/demo/pic2.jpg'
  result = inference_detector(model, img)
  savename = "./result/demo/pic_det2.png"
  show_result(img, result, model.CLASSES, score_thr=0.3, out_file=savename)

if __name__ == '__main__':
    main()