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
  config_file = "./configs/rscup/htc_sy.py"
  checkpoint_file = "./work_dirs/htc_sy/epoch_12.pth"
  # model = init_detector(config_file, checkpoint_file)
  # print(model.CLASSES)
  # pbar = tqdm(total=100)
  # def update(*a):
  #     pbar.update()
  # rets = []
  # for i in range(100):
  #   img = './result/demo/7.jpg'
  #   # torch.multiprocessing.spawn(inference_detector, args=(model, img), nprocs=2, join=True, daemon=False)
  #   rets.append(p.apply_async(inference_detector, args=(model, img), callback=update()))
  # for ret in rets:
  #     a = ret.get()
  num_processes = 4
  img = './result/demo/7.jpg'
  model = init_detector(config_file, checkpoint_file)
  model.cuda()
  # NOTE: this is required for the ``fork`` method to work
  model.share_memory()
  processes = []
  for rank in range(num_processes):
      p = mp.Process(target=inference_detector, args=(model, img))
      p.start()
      processes.append(p)
  for p in processes:
      p.join()


if __name__ == '__main__':
    main()