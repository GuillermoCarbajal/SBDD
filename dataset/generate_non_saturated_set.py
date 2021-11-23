import os
from skimage.io import imread, imsave
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.util import random_noise
from scipy import io
import argparse
from skimage.filters import gaussian
import time
import multiprocessing


def get_corrected_raw_names(image_fullname):

    # Read attributes
    # format: '%03d; %s;  %d; %s; "%s"\n', instance, name, whole(j)==0, crop, atr
    # format: Instance, part level (0 for objects), crop, class name, corrected_raw_name, list-of-attributes
    with open(image_fullname.replace('.jpg', '_atr.txt'), 'r') as f:
        text = f.readlines()
    corrected_raw_names, part_level = [], []
    for line in text:
        data = line.split(' # ')
        is_part = int(data[1]) > 0
        if not is_part:
            corrected_raw_names.append(data[4])

    return corrected_raw_names

def get_segmentation_info(image_fullname):
    '''
    :param image_fullname:
    :param sharp_image:
    :return:
        sharp_image_crop:
        B_crop:
        values_of_segmented_instances_in_crop:
        values_of_segmented_instances:
        corrected_raw_name: used to know whether is a person or a car
    '''

    seg_filename = image_fullname.replace('.jpg', '_seg.png')
    seg_img = np.array(Image.open(seg_filename))

    B = np.array(seg_img)[:, :, 2]
    values_of_segmented_instances = np.unique(B)

    corrected_raw_names = get_corrected_raw_names(image_fullname)

    values_of_segmented_instances_in_crop = np.unique(B)

    return B, values_of_segmented_instances_in_crop, values_of_segmented_instances, corrected_raw_names


def generate_blurry_sharp_pair(sharp_image_crop, kernels, masks_to_send, kernel_size=33, gray_scale=False,
                               gamma_correction=True, gamma_factor=2.2, augment_illumination=True,
                               jitter_illumination=1.0, poisson_noise=True, gaussian_noise=False, noise_std=0.01):

    K = kernel_size
    M,N,C = sharp_image_crop.shape

    if gray_scale:
        sharp_image_crop = (255*rgb2gray(sharp_image_crop)).astype(np.uint8)
        sharp_image_crop = sharp_image_crop[:,:,None]
        blurry_image = np.zeros((sharp_image_crop.shape[0] - K + 1, sharp_image_crop.shape[1] - K + 1, 1),
                                dtype=np.float32)
    else:
        blurry_image = np.zeros((sharp_image_crop.shape[0]-K+1,sharp_image_crop.shape[1]-K+1,3),  dtype=np.float32)

    if gamma_correction:
        sharp_image_crop = 255.0*( (sharp_image_crop/255.0)** gamma_factor)

    if augment_illumination:
        if sharp_image_crop.dtype == np.uint8:
            sharp_image_crop = rgb2hsv(sharp_image_crop)
            sharp_image_crop[:,:,2] *= (1+jitter_illumination*(np.random.rand()-0.5))
            sharp_image_crop = 255*hsv2rgb(sharp_image_crop)
        else:
            sharp_image_crop = rgb2hsv(sharp_image_crop/255)
            sharp_image_crop[:,:,2] *= (1+jitter_illumination*(np.random.rand()-0.5))
            sharp_image_crop = 255*hsv2rgb(sharp_image_crop)
        #sharp_image_crop = (1.5*sharp_image_crop).astype(np.uint8)


    # blurry image is generated
    for i in range(len(kernels)):
        kernel = kernels[i]
        blurry_k = fftconvolve(sharp_image_crop, kernel[::-1, ::-1, None],  mode='valid')
        #blurry_k = blurry_k[K // 2:-(K // 2), K // 2:-(K // 2)]
        mask = masks_to_send[:,:,i]
        #plt.imshow(mask)
    #plt.show()
        blurry_image += mask[:, :, None] * blurry_k

    if poisson_noise:
        max_value = np.max([255, blurry_image.max()])
        blurry_image = max_value * random_noise(blurry_image / max_value, mode="poisson", clip=False)


    if gaussian_noise:
        #blurry_image = 255 * random_noise(blurry_image / 255.0, mode="gaussian", var=noise_std**2, clip=False)
        blurry_image +=  255 * noise_std * np.random.randn(blurry_image.shape[0],blurry_image.shape[1],C)

    blurry_image[blurry_image<0] = 0
    blurry_image[blurry_image > 255] = 255

    if gamma_correction:
        sharp_image_crop = 255.0*( (sharp_image_crop/255.0)** (1.0/gamma_factor))
        blurry_image = 255.0 * ( (blurry_image / 255.0) ** (1.0 / gamma_factor))

    blurry_image = blurry_image.astype(np.float32)
    sharp_image_crop = sharp_image_crop.astype(np.float32)

    return blurry_image, sharp_image_crop

def process(n_filename):

    it, filename = n_filename
    filename = filename[:-1]
    img_name = filename.split('/')[-1]
    img_name, ext = img_name.split('.')
    image_fullname = os.path.join(ADE_DIR, filename)
    print('%d/%d :' % (it, len(files)), img_name)
    sharp_image = imread(image_fullname)
    saturated_contribution = np.zeros_like(sharp_image)

    if len(sharp_image.shape) == 3:
        M, N, C = sharp_image.shape
    else:
        M, N = sharp_image.shape
        C = 1
    min_size = np.min([M, N])

    if C > 1 and (min_size > min_img_size):
        # An interesting crop is chosen
        B, values_of_segmented_instances_in_crop, values_of_segmented_instances, corrected_raw_names = get_segmentation_info(
            image_fullname)

        for n in range(n_images):

            blurry_image_filename = os.path.join(output_dir, 'blurry', img_name + '_%d.png' % n)
            sharp_image_filename = os.path.join(output_dir, 'sharp', img_name + '_%d.png' % n)

            if (not os.path.exists(blurry_image_filename)) or (not os.path.exists(sharp_image_filename)):
                # structures to save mask and kernels.
                masks = []
                kernels = []

                # kernel = get_random_kernel()

                # Get the background kernel
                idx_kernel = np.random.randint(len(kernels_list))
                kernel_name = kernels_list[idx_kernel][0:-1].split('/')
                kernel = io.loadmat(os.path.join(Kernels_Dir, kernel_name[1]))['K']
                kernels.append(kernel)

                # Initializa mask as ones (everything is background)
                mask = np.ones((M, N), dtype=np.float32)
                # mask = mask[kernel_size // 2: -kernel_size // 2 + 1, kernel_size // 2: -kernel_size // 2 + 1]
                masks.append(mask)

                num_objects_blurred = 0
                for i, val in enumerate(values_of_segmented_instances_in_crop):
                    index = np.argwhere(values_of_segmented_instances == val)[0][0]
                    instance_label = corrected_raw_names[index - 1]
                    if instance_label in labels_to_blur and index > 0:  # primer elemento no tiene etiqueta
                        num_pix = np.sum(B == val)
                        if num_pix > min_objetc_area and num_objects_blurred < max_objects_to_blur:  # minima area para borronear el objeto
                            num_objects_blurred += 1
                            # kernel = get_random_kernel()
                            idx_kernel = np.random.randint(len(kernels_list))
                            kernel_name = kernels_list[idx_kernel][0:-1].split('/')
                            kernel = io.loadmat(os.path.join(Kernels_Dir, kernel_name[1]))['K']

                            mask = np.zeros_like(B, dtype=np.float32)  # nueva mascara
                            mask[B == val] = 1  # vale uno en el objeto
                            masks[0][
                                B == val] = 0  # se ponen a cero las posiciones del background que fueron sustituidas
                            # mask=mask[kernel_size//2: -kernel_size//2+1, kernel_size//2: -kernel_size//2+1]

                            kernels.append(kernel)
                            masks.append(mask)



                masks_to_send = np.zeros(
                    (sharp_image.shape[0] - kernel_size + 1, sharp_image.shape[1] - kernel_size + 1,
                     max_objects_to_blur + 1),
                    dtype=np.float32)
                kernels_to_send = np.zeros((kernel.shape[0], kernel.shape[1], max_objects_to_blur + 1),
                                           dtype=np.float32)

                # masks are convolved with kernels to smooth them and avoid discontinuities
                for i, mask in enumerate(masks):
                    kernel = kernels[i]
                    # mask = convolve(mask, kernel[::-1, ::-1])
                    mask = fftconvolve(mask, kernel[::-1, ::-1], mode='valid')
                    # mask = mask[kernel_size // 2:-(kernel_size // 2), kernel_size // 2:-(kernel_size // 2)]
                    masks_to_send[:, :, i] = mask
                    kernels_to_send[:, :, i] = kernel
                    # if len(masks) > 2:
                    #     imsave(f'kernel_{it}_{i}.png', (kernel - kernel.min()) / (kernel.max() - kernel.min()))

                # masks must be normalized because after filtering the sum is not one any more
                try:
                    masks_sum = np.sum(masks_to_send, axis=2)
                    masks_to_send = masks_to_send / (masks_sum[:, :, None] + 1e-6)
                except:
                    print('%d values in the masks summatory are zero' % np.sum(masks_sum == 0))

                blurry_image, sharp_image_crop = generate_blurry_sharp_pair(sharp_image, kernels, masks_to_send,
                                                                            kernel_size,
                                                                            gamma_factor=opt.gamma_factor,
                                                                            augment_illumination=augment_illumination)


                # imsave(os.path.join(output_dir, f'masks_{it}.png'), masks_to_send)
                imsave(blurry_image_filename,
                       np.clip(blurry_image, 0, 255).astype(np.uint8), check_contrast=False)
                imsave(sharp_image_filename, np.clip(
                    sharp_image_crop[kernel_size // 2:-(kernel_size // 2), kernel_size // 2:-(kernel_size // 2)], 0,
                    255).astype(np.uint8), check_contrast=False)
                # imsave(os.path.join(output_dir, f'kernels_{it}.png'), kernels_to_send)

def multiprocessing_func(n_file_name):
    time.sleep(2)
    n, file_name = n_file_name
    print('{}: filename is {}'.format(n, file_name ))
    process(n_file_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--kernels_dir', '-kd', type=str, help='kernels root folder', required=False,
                        default='/media/carbajal/OS/data/datasets/kernel_dataset/size_33_exp_1')
    parser.add_argument('--ade_dir', '-ad', type=str, help='ADE dir folder', required=False,
                        default='/media/carbajal/OS/data/datasets/ADE20K/ADE20K_2016_07_26/images')
    parser.add_argument('--kernels_list', '-kl', type=str, help='kernels list', required=False,
                        default='blur_kernels_train.txt')
    parser.add_argument('--ade_list', '-al', type=str, help='ADE list', required=False, default='non_saturated_set_list_small.txt')
    parser.add_argument('--K', type=int, help='kernel size', required=False, default=33)
    parser.add_argument('--min_img_size', type=int, help='min crop size', required=False, default=400)
    parser.add_argument('--min_object_area', type=int, help='min object area', required=False, default=400)
    parser.add_argument('--max_objects_to_blur', type=int, help='max objects to blur', required=False, default=5)
    parser.add_argument('--output_dir', '-o', type=str, help='path of the output dir', required=True)
    parser.add_argument('--gamma_factor', '-gf', type=float, default=2.2, help='gamma_factor', required=False)
    parser.add_argument('--augment_illumination', default=True, action='store_true',
                        help='whether to augment illumination')
    parser.add_argument('--n_images', type=int, help='number of blurry images per sharp image', required=False, default=2)

    opt = parser.parse_args()
    ADE_DIR = opt.ade_dir
    Kernels_Dir = opt.kernels_dir
    training_files_list = opt.ade_list
    kernels_images_list = opt.kernels_list
    output_dir = opt.output_dir
    min_img_size = opt.min_img_size
    kernel_size = opt.K
    labels_to_blur = ['person', 'car']
    min_objetc_area = opt.min_object_area
    max_objects_to_blur = opt.max_objects_to_blur
    augment_illumination = opt.augment_illumination
    gamma_factor = opt.gamma_factor
    n_images = 1
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'blurry')):
        os.makedirs(os.path.join(output_dir, 'blurry'))
    if not os.path.exists(os.path.join(output_dir, 'sharp')):
        os.makedirs(os.path.join(output_dir, 'sharp'))

    with open(training_files_list) as f:
        files = f.readlines()

    with open(kernels_images_list) as f:
        kernels_list = f.readlines()

    # files = files[:100]
    starttime = time.time()

    pool = multiprocessing.Pool(8)
    pool.map(multiprocessing_func, enumerate(files))
    pool.close()


    print('Time taken = {} seconds'.format(time.time() - starttime))
