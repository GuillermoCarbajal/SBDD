import argparse
import glob
import skimage
import os
import numpy as np
from multiprocessing import Pool
from functools import partial
import time
from evaluation_utils import evaluate_from_paths

parser = argparse.ArgumentParser(description='eval arg')
parser.add_argument('--blurry_dir', '-b', type=str, default='datasets/RealBlur/test/blur')
parser.add_argument('--restored_dir', '-r', type=str, default='datasets/RealBlur/test/restored')
parser.add_argument('--gt_dir', '-s',  type=str, default='datasets/RealBlur/test/gt')
parser.add_argument('--output_dir','-o', type=str, default='evaluation_results')
parser.add_argument('--resize_factor','-rf', type=int, default=1)

parser.add_argument('--core', type=int, default=4)
args = parser.parse_args()


def evaluation(blurry_dir, restored_dir, gt_dir , results_dir='parallel_output', resize_factor=1):

  if not os.path.exists(results_dir):
    os.mkdir(results_dir)

  blurry_list = glob.glob(blurry_dir + '/*.png')
  if len(blurry_list)==0:
      blurry_list = glob.glob(blurry_dir + '/*.jpg')
  blurry_list.sort()

  restored_list = glob.glob(restored_dir + '/*.png')
  if len(restored_list)==0:
      restored_list = glob.glob(restored_dir + '/*.jpg')
  restored_list.sort()

  gt_list = glob.glob(gt_dir + '/*.png')
  if len(gt_list)==0:
      gt_list = glob.glob(gt_dir + '/*.jpg')
  gt_list.sort()

  deblur_psnr_list = []
  deblur_ssim_list = []
  blur_psnr_list = []
  blur_ssim_list = []


  # N=250
  # blurry_list = blurry_list[:N]
  # restored_list = restored_list[:N]
  # gt_list = gt_list[:N]

  tic = time.time()

  processors = 8
  with Pool(processors) as p:
    output = p.map(partial(evaluate_from_paths, results_dir=results_dir, resize_factor=resize_factor), zip(blurry_list,restored_list,gt_list))

  f = open(os.path.join(results_dir, 'psnr_ssim.txt'), 'wt')
  for i, img in enumerate(blurry_list):
      blur_psnr = output[i][0]
      deblur_psnr = output[i][1]
      blur_ssim = output[i][2]
      deblur_ssim = output[i][3]
      blur_psnr_list.append(blur_psnr)
      deblur_psnr_list.append(deblur_psnr)
      blur_ssim_list.append(blur_ssim)
      deblur_ssim_list.append(deblur_ssim)
      f.write("%s blur_psnr=%5.5f, deblur_psnr=%5.5f, blur_ssim=%5.5f, deblur_ssim=%5.5f \n" % (img, blur_psnr, deblur_psnr, blur_ssim, deblur_ssim))

  toc = time.time()
  print('Time consumed: %f seconds' % (toc-tic))

  mean_deblur_psnr = np.mean(deblur_psnr_list)
  mean_deblur_ssim = np.mean(deblur_ssim_list)
  mean_blur_psnr = np.mean(blur_psnr_list)
  mean_blur_ssim = np.mean(blur_ssim_list)

  f2 = open(os.path.join(results_dir, 'result.txt'), 'wt')
  f2.write("deblur_psnr : %4.4f \n" % mean_deblur_psnr)
  f2.write("deblur_ssim : %4.4f \n" % mean_deblur_ssim)
  f2.write("blur_psnr : %4.4f \n" % mean_blur_psnr)
  f2.write("blur_ssim : %4.4f \n" % mean_blur_ssim)
  #f2.write("cnt : %4.4f \n" % cnt)
  f2.close()
  f.close()


if __name__ == '__main__':

  if skimage.__version__ != '0.17.2':
    print("please use skimage==0.17.2 and python3")
    exit()

  evaluation(args.blurry_dir, args.restored_dir,  args.gt_dir, args.output_dir, args.resize_factor)
