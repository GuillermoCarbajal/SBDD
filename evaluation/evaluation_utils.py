import argparse
import glob
import skimage
from skimage import io
from skimage import color
import os
import numpy as np
from skimage.metrics import structural_similarity
from skimage.util.dtype import dtype_range
from multiprocessing import Pool
from skimage import util
import cv2
from scipy.io import loadmat
import time
from functools import partial



def image_align(deblurred, gt):
  # this function is based on kohler evaluation code
  z = deblurred
  c = np.ones_like(z)
  x = gt

  zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching
  #zs = z.copy() #(np.sum(x * z) / np.sum(z * z)) * z  # simple intensity matching

  warp_mode = cv2.MOTION_HOMOGRAPHY
  warp_matrix = np.eye(3, 3, dtype=np.float32)

  # Specify the number of iterations.
  number_of_iterations = 100

  termination_eps = 0

  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              number_of_iterations, termination_eps)

  # Run the ECC algorithm. The results are stored in warp_matrix.
  try:
    (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)
  except:
    cc=15
    print("An error occured but it does not matter, homography is the identity")

  target_shape = x.shape
  shift = warp_matrix

  zr = cv2.warpPerspective(
    zs,
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_CUBIC+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_REFLECT)

  cr = cv2.warpPerspective(
    np.ones_like(zs, dtype='float32'),
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_NEAREST+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0)

  zr = zr * cr
  xr = x * cr

  return zr, xr, cr, shift


def compute_psnr(image_true, image_test, image_mask, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
  return 10 * np.log10((data_range ** 2) / err)


def im2uint8(image):
  image = np.clip(image * 255, 0, 255) + 0.5  # round color value
  image = image.astype('uint8')
  return image


def evaluate_from_paths(triplet, results_dir='parallel_output', compute_ssim = True, resize_factor=1):
    blurry_path, restored_path, gt_path = triplet
    print(blurry_path.split('/')[-1], restored_path.split('/')[-1], gt_path.split('/')[-1])
    #results_dir='parallel_output'
    #compute_ssim = False
    deblurred = io.imread(os.path.join(restored_path)).astype('float32') / 255
    if resize_factor > 1:
        M, N, C = deblurred.shape
        deblurred = resize(deblurred, (resize_factor*M, resize_factor*N))
    blurred = io.imread(os.path.join(blurry_path)).astype('float32') / 255
    # gt = loadmat(os.path.join(gt_dir, gt_list[j]))
    gt = io.imread(os.path.join(gt_path)).astype('float32') / 255

    return evaluate((blurred[:,:,:3], deblurred[:,:,:3], gt[:,:,:3]), output_name=restored_path.split('/')[-1],
                    results_dir=results_dir, compute_ssim=True, resize_factor=1)

def evaluate(triplet, output_name, results_dir='parallel_output', compute_ssim = True, resize_factor=1, save_image=True):
    '''
    :param triplet: np arrays of size (M,N,C)
    :param results_dir:
    :param compute_ssim:
    :param resize_factor:
    :return:
    '''
    blurred, deblurred, gt = triplet

    # B = int(16*resize_factor)
    # deblurred = deblurred[B:-B, B:-B, :]
    # blurred = blurred[B:-B, B:-B, :]
    # gt = gt[B:-B, B:-B, :]

    aligned_deblurred, aligned_xr1, cr1, shift = image_align(deblurred, gt)


    aligned_blurred = blurred
    aligned_xr2 = gt

    blur_ssim = 0
    deblur_ssim = 0
    if compute_ssim:
        # it is recomended by nah et al.
        deblur_ssim_pre, deblur_ssim_map = structural_similarity(aligned_xr1, aligned_deblurred, multichannel=True,
                                                                 gaussian_weights=True,
                                                                 use_sample_covariance=False, data_range=1.0,
                                                                 full=True)
        deblur_ssim_map = deblur_ssim_map * cr1

        r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
        win_size = 2 * r + 1
        pad = (win_size - 1) // 2
        deblur_ssim = deblur_ssim_map[pad:-pad, pad:-pad, :]
        crop_cr1 = cr1[pad:-pad, pad:-pad, :]
        deblur_ssim = deblur_ssim.sum(axis=0).sum(axis=0) / crop_cr1.sum(axis=0).sum(axis=0)
        deblur_ssim = np.mean(deblur_ssim)

        #blur_ssim = structural_similarity(aligned_xr2, aligned_blurred, multichannel=True, gaussian_weights=True,
        #                                  use_sample_covariance=False, data_range = 1.0, full=True)


        #print(deblur_ssim, deblur_ssim_pre)

    # only compute mse on valid region
    deblur_psnr = compute_psnr(aligned_xr1, aligned_deblurred, cr1, data_range=1)
    cr2 = np.ones_like(blurred, dtype='float32')
    blur_psnr = compute_psnr(aligned_xr2, aligned_blurred, cr2, data_range=1)

    if save_image:
        vis_image = np.concatenate([aligned_blurred, aligned_deblurred, aligned_xr1], axis=1)

        # vis_img_out_name = 'vis_%s_blur_%s_PSNR_%5.5f_%5.5f_SSIM_%5.5f_%5.5f.jpg' % (scene_name, img_name[:-4], blur_psnr, deblur_psnr, blur_ssim, deblur_ssim)
        vis_img_out_name = 'vis_blur_%s_PSNR_%5.5f_%5.5f.png' % (
        output_name, blur_psnr, deblur_psnr)
        vis_img_out_name = os.path.join(results_dir, vis_img_out_name)
        #io.imsave(deblur_out_name, im2uint8(deblurred))
        io.imsave(vis_img_out_name, im2uint8(vis_image))

    return blur_psnr, deblur_psnr, blur_ssim, deblur_ssim





