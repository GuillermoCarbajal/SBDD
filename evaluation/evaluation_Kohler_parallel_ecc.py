import argparse
import glob
import skimage
from skimage import io
import os
import numpy as np
from skimage.metrics import structural_similarity
from multiprocessing import Pool
import cv2
from scipy.io import loadmat
import time
from functools import partial
from skimage.transform import resize


parser = argparse.ArgumentParser(description='eval arg')
parser.add_argument('--blurry_dir','-b', type=str, default='KohlerDataset/BlurryImages')
parser.add_argument('--restored_dir','-r', type=str, default='KohlerDataset/RestoredImages')
parser.add_argument('--gt_dir','-s', type=str, default='KohlerDataset/groundTruthMatFiles/')
parser.add_argument('--output_dir','-o', type=str, default='Kohler_results')



parser.add_argument('--core', type=int, default=4)
args = parser.parse_args()

def image_align(deblurred, gt):
  # this function is based on kohler evaluation code
  if deblurred.shape != gt.shape:
      deblurred = resize(deblurred, (gt.shape[0], gt.shape[1]))
  z = deblurred
  c = np.ones_like(z)
  x = gt

  zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

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

def evaluate_from_paths(triplet, results_dir='kohler_parallel_output', compute_ssim = True):
    blurry_path, restored_path, gt_path = triplet
    print(triplet)
    deblurred = io.imread(restored_path).astype('float32') / 255
    blurred = io.imread(blurry_path).astype('float32') / 255
    gt = loadmat(gt_path)
    # gt = io.imread(os.path.join(gt_dir, gt_list[j])).astype('float32') / 255
    gt_images = gt['GroundTruth'].astype('float32') / 255

    N_gt = gt_images.shape[3]
    best_gt = None
    best_deblur_psnr = 0
    best_deblur_ssim = 0
    for n in range(N_gt):
        gt_n = gt_images[:, :, :, n]
        aligned_deblurred, aligned_xr1, cr1, shift = image_align(deblurred, gt_n)
        # aligned_blurred, aligned_xr2, cr2, shift = image_align(blurred, gt)

        aligned_blurred = blurred
        aligned_xr2 = gt_n

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

            # blur_ssim = structural_similarity(aligned_xr2, aligned_blurred, multichannel=True, gaussian_weights=True,
            #                                  use_sample_covariance=False, data_range = 1.0, full=True)

            blur_ssim = 0
            #print(deblur_ssim, deblur_ssim_pre)

        # only compute mse on valid region
        deblur_psnr = compute_psnr(aligned_xr1, aligned_deblurred, cr1, data_range=1)

        if (deblur_psnr > best_deblur_psnr and deblur_psnr < 60.0):
            print('%s: better psnr found for n=%d, psnr=%f ssim=%f' % (blurry_path, n, deblur_psnr,deblur_ssim))
            #print('--- shift:', shift)
            best_deblur_psnr = deblur_psnr
            best_deblur_ssim = deblur_ssim
            cr2 = np.ones_like(blurred, dtype='float32')
            blur_psnr = compute_psnr(aligned_xr2, aligned_blurred, cr2, data_range=1)
            vis_image = np.concatenate([aligned_blurred, aligned_deblurred, aligned_xr1], axis=1)


    deblur_out_name = os.path.join(results_dir, restored_path.split('/')[-1])
    # vis_img_out_name = 'vis_%s_blur_%s_PSNR_%5.5f_%5.5f_SSIM_%5.5f_%5.5f.jpg' % (scene_name, img_name[:-4], blur_psnr, deblur_psnr, blur_ssim, deblur_ssim)
    vis_img_out_name = 'vis_blur_%s_PSNR_%5.5f_%5.5f_SSIM_%5.5f.jpg' % (restored_path.split('/')[-1], blur_psnr, best_deblur_psnr, best_deblur_ssim)
    vis_img_out_name = os.path.join(results_dir, vis_img_out_name)
    io.imsave(deblur_out_name, im2uint8(deblurred))
    io.imsave(vis_img_out_name, im2uint8(vis_image))
    # save GT pair
    io.imsave(restored_path.split('/')[-1] + '_GT.png', im2uint8(vis_image[:,1600:,:]))

    return blur_psnr, best_deblur_psnr,best_deblur_ssim

def evaluation_Kohler(blurry_dir, restored_dir, gt_dir , results_dir='kohler_parallel_output'):


  if not os.path.exists(results_dir):
    os.mkdir(results_dir)

  blurry_list = glob.glob(blurry_dir + '/*.png')
  blurry_list.sort()
  #print(blurry_list)
  restored_list = [restored_dir + '/' + s.split('/')[-1] for s in blurry_list]
  restored_list.sort()
  gt_list = glob.glob(gt_dir + '/*.mat')
  gt_list.sort()
  cnt = 0
  deblur_psnr_list = []
  deblur_ssim_list = []
  blur_psnr_list = []
  blur_ssim_list = []
  f = open(os.path.join(results_dir, 'psnr.txt'), 'wt')

  tic = time.time()

  processors = 4
  with Pool(processors) as p:
    output = p.map(partial(evaluate_from_paths, results_dir=results_dir), zip(blurry_list,restored_list,gt_list))

  f = open(os.path.join(results_dir, 'psnr.txt'), 'wt')
  for i, img in enumerate(blurry_list):
      blur_psnr = output[i][0]
      deblur_psnr = output[i][1]
      deblur_ssim = output[i][2]
      blur_psnr_list.append(blur_psnr)
      deblur_psnr_list.append(deblur_psnr)
      deblur_ssim_list.append(deblur_ssim)
      f.write("%s blur_psnr=%5.5f, deblur_psnr=%5.5f, deblur_ssim=%5.5f  \n" % (img, blur_psnr, deblur_psnr, deblur_ssim))

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
  f2.write("cnt : %4.4f \n" % cnt)
  f2.close()
  f.close()

if __name__ == '__main__':

  if skimage.__version__ != '0.17.2':
    print("please use skimage==0.17.2 and python3")
    exit()

  #if cv2.__version__ != '4.2.0':
  #  print("please use cv2==4.2.0.32 and python3")
  #  exit()

  evaluation_Kohler(args.blurry_dir, args.restored_dir,  args.gt_dir, args.output_dir)
