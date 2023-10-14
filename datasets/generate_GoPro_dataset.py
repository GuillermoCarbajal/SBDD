import os
from skimage.io import imread, imsave
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.util import random_noise
from skimage.morphology import closing, square
from skimage.filters import gaussian
from scipy import io
import argparse
import time
import multiprocessing

from matplotlib import pyplot as plt
import json


def lin2rgb_exp(photons, a=7):
    y = -  (np.exp(a*(-photons))-1 ) 
    return y
 
def rgb2lin_exp(img, a=7,b=1):    
    log_x = np.log( -img + 1e-9 + 1   )/b
    photons = -log_x /a
    return photons

def get_segmentation_mask(metadata_name, score_th=0.9, mob=2, min_area = 400):
    '''
    Entrada:
        metadata_name: .npz metadata file
        score_th: minimum score to choose an instance
        mob: maximum number of objects to blur
        min_area: minimum number of pixels to choose a mask
        
    Salida:
        segmentation mask: each mask is represented by a different number
    '''
    
    
    data = np.load(metadata_name, allow_pickle=False)
    instances_masks = data['pred_masks']
    N, H, W = instances_masks.shape
    mask_areas = np.count_nonzero(instances_masks, axis=(1,2))
    instances_scores = data['scores']
    instances_classes = data['pred_classes']
    
    # boolean array indicating whether the instance is of interest or not 
    is_class_of_interest = np.array([ class_names[c+1] in classes_of_interest for c in instances_classes])
    
    # boolean arrays to filter by score and area
    high_scores = instances_scores > score_th  # ()
    high_areas = mask_areas > min_area
    
    chosen_indices = np.logical_and(high_scores, high_areas)  # filter by score and area
    chosen_indices = np.logical_and(chosen_indices, is_class_of_interest) # filter by score, area and category
    
    Nf = np.sum(chosen_indices)
    print(f'images has {N} instances, {Nf} survived after filtering by confidence, area and category')
    chosen_masks_areas = mask_areas[chosen_indices]
    chosen_masks_scores = instances_scores[chosen_indices]
    print('The scores are ', chosen_masks_scores)
    
    # transform from (N) boolean array of to (Nf) integer array with the chosen instances ids
    chosen_instances_ids = [ int(i) for i, chosen in enumerate(chosen_indices) if chosen ]
    chosen_masks = instances_masks[chosen_instances_ids]
    
    
    # chosen masks are ordered by score
    order = np.argsort(chosen_masks_scores)[::-1]
    # a maximum of mob2 instances are returned
    instances_returned = order[:mob]
    print('According to accuracy the order is', order, '. ',instances_returned, 'are returned')
    

    segmentation_mask = []
    for i, n in enumerate(instances_returned):
        print(i, 'score=' ,chosen_masks_scores[n]  ,'area = ',chosen_masks_areas[n])
        mask_n = chosen_masks[n]
        segmentation_mask.append(mask_n)
    
    return segmentation_mask

def compute_saturation_mask(sharp_image, threshold=255):
    M,N,C = sharp_image.shape
    saturation_mask = 255*(sharp_image>=threshold)   
                                                               
    for c in range(C):
        saturation_mask[:, :, c] = gaussian(closing(saturation_mask[:, :, c], square(3)).astype(np.float32), sigma=1)

    num_saturated_pixels = np.sum(saturation_mask > 0)
    saturation_percentage = np.float(num_saturated_pixels) / (M * N)
    print('number of saturated pixels simulated: %d, percentage = %.04f ' % (num_saturated_pixels, saturation_percentage))

    return saturation_mask, saturation_percentage

def generate_blurry_sharp_pair(sharp_image_crop, kernels, masks_to_send, kernel_size=33, gray_scale=False,
                               gamma_correction=True, gamma_factor=2.2, augment_illumination=True,
                               jitter_illumination=1.0, poisson_noise=False, gaussian_noise=False, noise_std=0.01,
                               random_whyte_streaks=False, prob_streaks=0.5, min_streaks=10, max_streaks=100,
                               min_sat_increment=0,max_sat_increment=2000, saturation_streaks=False,
                               sat_augmentation_in_blurry=False, exponential_factor=7,exponential_crf=False):

    K = kernel_size
    M,N,C = sharp_image_crop.shape
    if gray_scale:
        sharp_image_crop = (255*rgb2gray(sharp_image_crop)).astype(np.uint8)
        sharp_image_crop = sharp_image_crop[:,:,None]
        blurry_image = np.zeros((sharp_image_crop.shape[0] - K + 1, sharp_image_crop.shape[1] - K + 1, 1),
                                dtype=np.float32)
    else:
        blurry_image = np.zeros((sharp_image_crop.shape[0]-K+1,sharp_image_crop.shape[1]-K+1,3),  dtype=np.float32)

    if exponential_crf:
        print('rgb2lin_exp: before  ', sharp_image_crop.min(),sharp_image_crop.max())
        sharp_image_crop = 255.0 * rgb2lin_exp(sharp_image_crop/255.0, a=exponential_factor)
        print('rgb2lin_exp: after ', sharp_image_crop.min(),sharp_image_crop.max())
    elif gamma_correction:
        sharp_image_crop = 255.0*( (sharp_image_crop/255.0)** gamma_factor)
    
    if augment_illumination:
         if sharp_image_crop.dtype == np.uint8:
             sharp_image_crop = rgb2hsv(sharp_image_crop)
             sharp_image_crop[:,:,2] *= (1+2*jitter_illumination*(np.random.rand())) 
             sharp_image_crop = 255*hsv2rgb(sharp_image_crop)
         else:
             sharp_image_crop = rgb2hsv(sharp_image_crop/255)
             sharp_image_crop[:,:,2] *= (1+2*jitter_illumination*(np.random.rand())) 
             sharp_image_crop = 255*hsv2rgb(sharp_image_crop)
        
        
    saturation_mask, sat_percentage = compute_saturation_mask(sharp_image_crop, threshold=255)
    if sat_percentage < 1.0/1000:
        saturation_mask = None
        
        
    if random_whyte_streaks and np.random.rand()<prob_streaks:
        for _ in range(np.random.randint(min_streaks, max_streaks)):
            i = np.random.randint(K,sharp_image_crop.shape[0]-K)
            j = np.random.randint(K,sharp_image_crop.shape[1]-K)
            ri = np.random.randint(4)  #4
            rj = np.random.randint(5)  #5
            spike = min_sat_increment + (max_sat_increment-min_sat_increment)*np.random.rand() 
            sharp_image_crop[i-ri:i+ri,j-rj:j+rj,:] = sharp_image_crop[i-ri:i+ri,j-rj:j+rj,:] + spike
                   
    if (saturation_mask is not None) and saturation_streaks:
        mask = saturation_mask==255
        print('Num pixels in masks: ', mask.sum())
        sharp_image_crop[mask] *= (1+ 4*np.random.rand())

        
    # blurry image is generated
    for i in range(len(kernels)):
        kernel = kernels[i]
        blurry_k = fftconvolve(sharp_image_crop, kernel[::-1, ::-1, None],  mode='valid')
        mask = masks_to_send[:,:,i]
        blurry_image += mask[:, :, None] * blurry_k
        

    if poisson_noise:
        max_value = np.max([255, blurry_image.max()])
        blurry_image = max_value * random_noise(blurry_image / max_value, mode="poisson", clip=False)


    if gaussian_noise:
        #blurry_image = 255 * random_noise(blurry_image / 255.0, mode="gaussian", var=noise_std**2, clip=False)
        #blurry_image +=  255*noise_std * np.random.randn(blurry_image.shape[0],blurry_image.shape[1],C)
        blurry_image +=  blurry_image*noise_std * np.random.randn(blurry_image.shape[0],blurry_image.shape[1],C)

    if (saturation_mask is not None) and (sat_augmentation_in_blurry):

        blurry_mask = np.zeros((sharp_image_crop.shape[0]-K+1,sharp_image_crop.shape[1]-K+1,3),  dtype=np.float32)  
      
        for i in range(len(kernels)):
            kernel = kernels[i]
            blurry_mask_k = fftconvolve(saturation_mask, kernel[::-1, ::-1, None],  mode='valid')
            mask = masks_to_send[:,:,i]
            blurry_mask  += mask[:, :, None] * blurry_mask_k
            
        blurry_image = blurry_image + blurry_mask*(4*np.random.rand())
    


    if exponential_crf:
        sharp_image_crop = 255*lin2rgb_exp(sharp_image_crop/255.0, a=exponential_factor)
        blurry_image = 255*lin2rgb_exp(blurry_image/255.0, a=exponential_factor)
     
    blurry_image[blurry_image<0] = 0
    blurry_image[blurry_image > 255] = 255    
        
    if gamma_correction:
    
        sharp_image_crop = 255.0*( (sharp_image_crop/255.0)** (1.0/gamma_factor))
        blurry_image = 255.0 * ( (blurry_image / 255.0) ** (1.0 / gamma_factor))
    


        
    blurry_image = blurry_image.astype(np.float32)
    sharp_image_crop = sharp_image_crop.astype(np.float32)

    return blurry_image, sharp_image_crop, saturation_mask

def process(file_name):


    sharp_image = np.array(Image.open(os.path.join(opt.gopro_sharp_dir, file_name)))
    img_name, ext = file_name.split('.')
    
    if random_masks:
        a = np.random.randint(len(training_files_list))
        mask_name, ext = training_files_list[a].split('.')
        metadata_name = os.path.join(segmentation_folder, mask_name + '.npz')
    else:
        metadata_name = os.path.join(segmentation_folder, img_name + '.npz')
        
    segmentation_masks = []
    if max_objects_to_blur>0:
        segmentation_masks = get_segmentation_mask(metadata_name, score_th=score_threshold, mob=max_objects_to_blur, min_area=min_objetc_area)

    if len(sharp_image.shape) == 3:
        M, N, C = sharp_image.shape
    else:
        M, N = sharp_image.shape
        C = 1
    min_size = np.min([M, N])


    if C > 1 and (min_size > min_img_size):

        for n in range(n_images):

            blurry_image_filename = os.path.join(output_dir, 'blurry', img_name + '_%d.jpg' % n)
            sharp_image_filename = os.path.join(output_dir, 'sharp', img_name + '_%d.jpg' % n)
            
            if (not os.path.exists(blurry_image_filename)) or (not os.path.exists(sharp_image_filename)):
                # structures to save mask and kernels.
                masks = []
                kernels = []

            
                # Get the background kernel
                idx_kernel = np.random.randint(len(kernels_list))
                kernel_name = kernels_list[idx_kernel][0:-1].split('/')
                kernel = io.loadmat(os.path.join(kernels_folder, kernel_name[1]))['K']
                kernels.append(kernel)

                # Initializa mask as ones (everything is background)
                mask = np.ones((M, N), dtype=np.float32)
                masks.append(mask)
                
                for object_mask in segmentation_masks:
    
                    idx_kernel = np.random.randint(len(kernels_list))
                    kernel_name = kernels_list[idx_kernel][0:-1].split('/')
                    kernel = io.loadmat(os.path.join(kernels_folder, kernel_name[1]))['K']

                    kernels.append(kernel)
                    masks.append(object_mask.astype(np.float32))
                    masks[0][object_mask == 1] = 0  # replaced background positions are set to zero 
                    
                for k, m in zip(kernels, masks):
                    print('image_name = %s; n=%d; img shape = %s; num pixels kernel = %f; num pixels masks = %f ' % (file_name, n, sharp_image.shape, (k>0).sum(), (m==1).sum()))

                masks_to_send = np.zeros(
                    (sharp_image.shape[0] - kernel_size + 1, sharp_image.shape[1] - kernel_size + 1,
                     max_objects_to_blur + 1),
                    dtype=np.float32)
                kernels_to_send = np.zeros((kernel.shape[0], kernel.shape[1], max_objects_to_blur + 1),
                                           dtype=np.float32)

                # masks are convolved with kernels to smooth them and avoid discontinuities
                for i, mask in enumerate(masks):
                    
                    kernel = kernels[i]
                    mask = fftconvolve(mask, kernel[::-1, ::-1], mode='valid') 
                    
                    masks_to_send[:, :, i] = mask
                    kernels_to_send[:, :, i] = kernel

                # masks must be normalized because after filtering the sum is not one any more
                try:
                    masks_sum = np.sum(masks_to_send, axis=2)
                    masks_to_send = masks_to_send / (masks_sum[:, :, None] + 1e-6)
                except:
                    print('%d values in the masks summatory are zero' % np.sum(masks_sum == 0))
                    
                blurry_image, sharp_image_crop, sat_mask = generate_blurry_sharp_pair(sharp_image, kernels, masks_to_send,
                                                                            kernel_size,
                                                                            gamma_factor=opt.gamma_factor,
                                                                            augment_illumination=augment_illumination,
                                                                            random_whyte_streaks=opt.random_whyte_streaks,
                                                                            gaussian_noise=opt.gaussian_noise,
                                                                            noise_std=opt.noise_std,
                                                                            prob_streaks=opt.prob_streaks,
                                                                            max_sat_increment=opt.max_sat_increment,
                                                                            min_streaks=opt.min_streaks,
                                                                            max_streaks=opt.max_streaks,
                                                                            sat_augmentation_in_blurry=opt.sat_augmentation_in_blurry,
                                                                            saturation_streaks=opt.saturation_streaks,
                                                                            exponential_factor=opt.exponential_factor,
                                                                            exponential_crf=opt.exponential_crf)


                imsave(blurry_image_filename,
                       np.clip(blurry_image, 0, 255).astype(np.uint8), check_contrast=False)
                imsave(sharp_image_filename, np.clip(
                    sharp_image_crop[kernel_size // 2:-(kernel_size // 2), kernel_size // 2:-(kernel_size // 2)], 0,
                    255).astype(np.uint8), check_contrast=False)
                
def multiprocessing_func(n_file_name):
    time.sleep(2)
    #n, file_name = n_file_name
    #print('{}: filename is {}'.format(n, file_name ))
    process(n_file_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--kernels_dir', '-kd', type=str, help='kernels root folder', required=False,
                        default='/media/carbajal/OS/data/datasets/kernel_dataset/size_33_exp_1/train')
    parser.add_argument('--gopro_sharp_dir', '-ad', type=str, help='GoPro sharp images folder', required=False,
                        default='/media/carbajal/OS/data/datasets/GOPRO_Large/prepared-data/sharp/train')
    parser.add_argument('--kernels_list', '-kl', type=str, help='kernels list', required=False,
                        default='datasets/blur_kernels_train.txt')
    parser.add_argument('--gopro_segmentation_dir', '-sd', type=str, help='GoPro segmentation folder', required=False,
                        default='/media/carbajal/OS/data/datasets/GoPro_detectron2_segmentation')
    parser.add_argument('--K', type=int, help='kernel size', required=False, default=65)
    parser.add_argument('--min_img_size', type=int, help='min crop size', required=False, default=400)
    parser.add_argument('--min_object_area', type=int, help='min object area', required=False, default=4000)
    parser.add_argument('--max_objects_to_blur', type=int, help='max objects to blur', required=False, default=5)
    parser.add_argument('--score_threshold', type=float, help='confidence score threshold', required=False, default=0.9)
    parser.add_argument('--output_dir', '-o', type=str, help='path of the output dir', required=True)
    parser.add_argument('--gamma_factor', '-gf', type=float, default=2.2, help='gamma_factor', required=False)
    parser.add_argument('--augment_illumination', default=False, action='store_true',
                        help='whether to augment illumination')
    parser.add_argument('--exponential_factor', '-ef', type=float, default=7, help='exponential_factor', required=False)
    parser.add_argument('--exponential_crf', default=False, action='store_true',
                        help='whether to use exponential crf')
    parser.add_argument('--n_images', type=int, help='number of blurry images per sharp image', required=False, default=1)
    parser.add_argument('--random_masks', default=False, action='store_true',
                        help='whether to use random masks')
    parser.add_argument('--random_whyte_streaks', action='store_true', default=False,help='whether to generate random whyte streaks')
    parser.add_argument('--gaussian_noise', action='store_true', default=False,help='whether to add gaussian noise')
    parser.add_argument('--noise_std', type=float, default=0,help='noise std')
    parser.add_argument('--prob_streaks', '-ps', type=float, default=0.5, help='streaks probability', required=False)
    parser.add_argument('--max_sat_increment', type=int, help='max_sat_increment', required=False, default=2000)
    parser.add_argument('--min_streaks', type=int, help='min_streaks', required=False, default=10)
    parser.add_argument('--max_streaks', type=int, help='max_streaks', required=False, default=100)
    parser.add_argument('--saturation_mask', action='store_true', default=False,help='whether to generate a saturation mask from sharp image')
    parser.add_argument('--saturation_streaks', action='store_true', default=False,help='generate streaks from saturation mask')
    parser.add_argument('--sat_augmentation_in_blurry', action='store_true', default=False,help='generate saturation augmentation in blurry image instead of sharp')


    opt = parser.parse_args()
    kernels_folder = opt.kernels_dir
    segmentation_folder = opt.gopro_segmentation_dir
    kernels_images_list = opt.kernels_list
    output_dir = opt.output_dir
    min_img_size = opt.min_img_size
    kernel_size = opt.K
    min_objetc_area = opt.min_object_area
    max_objects_to_blur = opt.max_objects_to_blur
    score_threshold = opt.score_threshold
    augment_illumination = opt.augment_illumination
    gamma_factor = opt.gamma_factor
    random_masks = opt.random_masks
    n_images = opt.n_images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'blurry')):
        os.makedirs(os.path.join(output_dir, 'blurry'))
    if not os.path.exists(os.path.join(output_dir, 'sharp')):
        os.makedirs(os.path.join(output_dir, 'sharp'))
    #if not os.path.exists(os.path.join(output_dir, 'masks')):
    #    os.makedirs(os.path.join(output_dir, 'masks'))
        
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)
        
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                    'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                    'teddy bear', 'hair drier', 'toothbrush']

    classes_of_interest = class_names # all
    
    training_files_list = os.listdir(opt.gopro_sharp_dir)
    
    print('Number of training images: %d' % len(training_files_list))

    with open(kernels_images_list) as f:
        kernels_list = f.readlines()

    # files = files[:100]
    starttime = time.time()

    pool = multiprocessing.Pool(1)
    pool.map(multiprocessing_func, training_files_list)
    pool.close()


    print()
    print('Time taken = {} seconds'.format(time.time() - starttime))
