import os
from skimage.io import imread, imsave
import numpy as np
from PIL import Image
#from scipy.ndimage import convolve
from scipy.signal import fftconvolve
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.util import random_noise
from scipy import io
import argparse
from skimage.morphology import closing, square
from skimage.filters import gaussian
import time
import multiprocessing
import pickle as pkl
import utils_ade20k



def generate_blurry_sharp_pair(sharp_image_crop, kernels, masks_to_send, kernel_size=33, gray_scale=False,
                               gamma_correction=True, gamma_factor=2.2, augment_illumination=True,
                               jitter_illumination=1.0, poisson_noise=False, saturation_mask=None, gaussian_noise=False, noise_std=0.01):

    K = kernel_size

    if gray_scale:
        sharp_image_crop = (255*rgb2gray(sharp_image_crop)).astype(np.uint8)
        sharp_image_crop = sharp_image_crop[:,:,None]
        blurry_image = np.zeros((sharp_image_crop.shape[0] - K + 1, sharp_image_crop.shape[1] - K + 1, 1),
                                dtype=np.float32)
    else:
        blurry_image = np.zeros((sharp_image_crop.shape[0]-K+1,sharp_image_crop.shape[1]-K+1,3),  dtype=np.float32)

    if gamma_correction:
        sharp_image_crop = 255.0*( (sharp_image_crop/255.0)** gamma_factor)

    M,N,C = sharp_image_crop.shape
    for c in range(C):
        saturation_mask[:, :, c] = gaussian(closing(saturation_mask[:, :, c], square(3)).astype(np.float32), sigma=1)

    if augment_illumination:
        if sharp_image_crop.dtype == np.uint8:
            sharp_image_crop_sat = rgb2hsv(sharp_image_crop)
            sharp_image_crop_non_sat = rgb2hsv(sharp_image_crop)
        else:
            sharp_image_crop_sat = rgb2hsv(sharp_image_crop/255)
            sharp_image_crop_non_sat = rgb2hsv(sharp_image_crop/255)

        random_low = 0.25*jitter_illumination * np.random.rand() #0.75*jitter_illumination * np.random.rand()
        random_high = jitter_illumination * np.random.rand()
        #print('random low: ', random_low, ' random_high: ', random_high)
        sharp_image_crop_non_sat[:, :, 2] *= (0.25 + random_low) # (0.05 + random_low) #(0.25 + random_low)
        sharp_image_crop_sat[:,:,2]*=  (0.25 + random_low  + random_high) #(0.75 + random_high) #(0.25 + random_low  + random_high)
        sharp_image_crop = 255*hsv2rgb(sharp_image_crop_sat * saturation_mask) + 255*hsv2rgb(sharp_image_crop_non_sat * (1-saturation_mask))
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
        blurry_image +=  255*noise_std * np.random.randn(blurry_image.shape[0],blurry_image.shape[1],C)

    blurry_image[blurry_image < 0] = 0
    blurry_image[blurry_image > 255] = 255

    blurry_image = blurry_image.astype(np.float32)
    sharp_image_crop = sharp_image_crop.astype(np.float32)

    if gamma_correction:
        sharp_image_crop = 255.0*( (sharp_image_crop/255.0)** (1.0/gamma_factor))
        blurry_image = 255.0 * ((blurry_image / 255.0) ** (1.0 / gamma_factor))

    return blurry_image, sharp_image_crop

def process(n_file):

    filename = index_ade20k['filename'][n_file]
    img_name = filename.split('/')[-1]
    img_name, ext = img_name.split('.')
    full_file_name = '{}/{}'.format(index_ade20k['folder'][n_file], index_ade20k['filename'][n_file])
    image_fullname = os.path.join(ADE_DIR, full_file_name)
    sharp_image = imread(image_fullname)

    if len(sharp_image.shape) == 3:
        M, N, C = sharp_image.shape
    else:
        M, N = sharp_image.shape
        C = 1
    min_size = np.min([M, N])

    if C > 1 and (min_size > min_img_size):
        #B, values_of_segmented_instances_in_crop, values_of_segmented_instances, corrected_raw_names = get_segmentation_info(
        #    image_fullname)

        info = utils_ade20k.loadAde20K(image_fullname)
        B =  imread(info['segm_name'])
        objs_info = info['objects']
        instances_indices = objs_info['instancendx']
        instances_classes = objs_info['class']
        instances_corrected_raw_name = objs_info['corrected_raw_name']
        for n in range(n_images):

            light_masks_filename = os.path.join(output_dir, 'light_masks', img_name + '_%d.jpg' % n)
            blurry_filename = os.path.join(output_dir, 'blurry', img_name + '_%d.jpg' % n)
            sharp_filename = os.path.join(output_dir, 'sharp', img_name + '_%d.jpg' % n)

            if not os.path.exists(light_masks_filename) or not os.path.exists(blurry_filename)  or not os.path.exists(sharp_filename) :
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

                saturation_mask = np.zeros_like(sharp_image, dtype=np.float32)
                num_objects_blurred = 0
                for obj_id, obj_label in zip(instances_indices,instances_classes) :

                    if obj_label in labels_to_blur:  # primer elemento no tiene etiqueta
                        file_instance = '{}/instance_{:03}_{}'.format(
                            image_fullname.replace('.jpg', ''), obj_id, filename.replace('.jpg', '.png'))
                        #print(file_instance)
                        B = imread(file_instance)
                        B[B < 255] = 0
                        num_pix = np.sum(B == 255)
                        if num_pix > min_objetc_area and num_objects_blurred < max_objects_to_blur:  # minima area para borronear el objeto
                            num_objects_blurred += 1
                            # kernel = get_random_kernel()
                            idx_kernel = np.random.randint(len(kernels_list))
                            kernel_name = kernels_list[idx_kernel][0:-1].split('/')
                            kernel = io.loadmat(os.path.join(Kernels_Dir, kernel_name[1]))['K']

                            mask = B.copy()
                            # The 0 index in seg_mask corresponds to background (not annotated) pixels
                            masks[0] [mask>0] = 0    # se ponen a cero las posiciones del background que fueron sustituidas

                            #mask = np.zeros_like(B, dtype=np.float32)  # nueva mascara
                            #mask[B == obj_id] = 1  # vale uno en el objeto
                            #masks[0][
                            #    B == obj_id] = 0  # se ponen a cero las posiciones del background que fueron sustituidas
                            # mask=mask[kernel_size//2: -kernel_size//2+1, kernel_size//2: -kernel_size//2+1]

                            kernels.append(kernel)
                            masks.append(mask)
                    elif obj_label in light_labels:
                        file_instance = '{}/{}/instance_{:03}_{}'.format(
                            ADE_DIR, full_file_name.replace('.jpg', ''), obj_id, filename.replace('.jpg', '.png'))
                        #print(file_instance)
                        B = imread(file_instance)
                        light_mask = B == 255
                        candidate_region = sharp_image > 250
                        candidate_region = np.max(candidate_region, axis=2)

                        sat_region_values = np.logical_and(light_mask, candidate_region)
                        saturation_mask[sat_region_values] = 1

                num_saturated_pixels = np.sum(saturation_mask > 0)
                saturation_percentage = np.float(num_saturated_pixels) / (M * N)
                print('%s: number of saturated pixels simulated: %d, percentage = %.04f ' % (img_name,
                    num_saturated_pixels, saturation_percentage))

                if saturation_percentage > 1.0 / 1000:
                    # saturation_mask = gaussian(saturation_mask)
                    # masks[0]=convolve(masks[0], kernels[0][::-1])
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
                                                                                augment_illumination=augment_illumination,
                                                                                saturation_mask=saturation_mask)

                    imsave(light_masks_filename,
                           (255*saturation_mask[kernel_size // 2:-(kernel_size // 2),
                           kernel_size // 2:-(kernel_size // 2)]).astype(np.uint8), check_contrast=False)
                    # imsave(os.path.join(output_dir, f'masks_{it}.png'), masks_to_send)
                    imsave(blurry_filename,
                           np.clip(blurry_image, 0, 255).astype(np.uint8), check_contrast=False)
                    imsave(sharp_filename, np.clip(
                        sharp_image_crop[kernel_size // 2:-(kernel_size // 2), kernel_size // 2:-(kernel_size // 2)], 0,
                        255).astype(np.uint8), check_contrast=False)
                    # imsave(os.path.join(output_dir, f'kernels_{it}.png'), kernels_to_send)
                    print('%s utilized' % img_name)
                else:
                    print('%s discarded' % img_name)

def multiprocessing_func(n_file):
    time.sleep(2)
    process(n_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--kernels_dir', '-kd', type=str, help='kernels root folder', required=False,
                        default='/data_ssd/users/carbajal/kernel_dataset/size_33_exp_1/train')
    parser.add_argument('--ade_dir', '-ad', type=str, help='ADE dir folder', required=False,
                        default='/data_ssd/users/carbajal/ADE20K_new/carbajal_776b3c5f')
    parser.add_argument('--kernels_list', '-kl', type=str, help='kernels list', required=False, default='blur_kernels_train.txt')
    parser.add_argument('--ade_index', '-al', type=str, help='ADE list', required=False,
                        default='/data_ssd/users/carbajal/ADE20K_new/carbajal_776b3c5f/ADE20K_2021_17_01/index_ade20k.pkl')
    parser.add_argument('--K', type=int, help='kernel size', required=False, default=33)
    parser.add_argument('--min_img_size', type=int, help='min crop size', required=False, default=400)
    parser.add_argument('--min_object_area', type=int, help='min object area', required=False, default=400)
    parser.add_argument('--max_objects_to_blur', type=int, help='max objects to blur', required=False, default=5)
    parser.add_argument('--output_dir', '-o', type=str, help='path of the output dir', required=True)
    parser.add_argument('--gamma_factor', '-gf', type=float, default=1.0, help='gamma_factor', required=False)
    parser.add_argument('--augment_illumination', default=True, action='store_true',
                        help='whether to augment illumination')
    parser.add_argument('--n_images', type=int, help='number of blurry images per sharp image', required=False, default=2)

    opt = parser.parse_args()
    ADE_DIR = opt.ade_dir
    Kernels_Dir = opt.kernels_dir
    ade_index = opt.ade_index
    kernels_images_list = opt.kernels_list
    output_dir = opt.output_dir
    min_img_size = opt.min_img_size
    kernel_size = opt.K
    labels_to_blur = ['car']
    # light_labels = ['bulb', 'ceiling recessed light', 'ceiling spotlight', 'ceiling spotlights', 'christmas lights',
    #                 'flashlight', 'floor light', 'floor recessed light', 'floor spotlight', 'light troffer',
    #                 'lighthouse', 'lighting', 'night light', 'spotlight', 'spotlights', 'street light', 'street lights',
    #                 'wall recessed light', 'wall spotlight', 'wall spotlights']
    light_labels = ['bulb','bulbs','ceiling spotlight','ceiling recessed light','christmas lights','flush mount light', 'light troffer', 'semi-flush mount lights',
                      'skylight', 'spotlight','spotlights','street light', 'traffic light', 'wall recessed light','wall spotlight', 'wall spotlights',
                      'pendant lamp', 'table lamp']
    min_objetc_area = opt.min_object_area
    max_objects_to_blur = opt.max_objects_to_blur
    augment_illumination = opt.augment_illumination
    gamma_factor = opt.gamma_factor
    n_images = opt.n_images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'blurry')):
        os.makedirs(os.path.join(output_dir, 'blurry'))
    if not os.path.exists(os.path.join(output_dir, 'sharp')):
        os.makedirs(os.path.join(output_dir, 'sharp'))
    if not os.path.exists(os.path.join(output_dir, 'light_masks')):
        os.makedirs(os.path.join(output_dir, 'light_masks'))

    #with open(training_files_list) as f:
    #    files = f.readlines()

    with open(ade_index, 'rb') as f:
        index_ade20k = pkl.load(f)

    print("File loaded, description of the attributes:")
    print('--------------------------------------------')
    for attribute_name, desc in index_ade20k['description'].items():
        print('* {}: {}'.format(attribute_name, desc))
    print('--------------------------------------------\n')


    def get_person_and_licence_plate_ids(index_ade20k):

        person_ids = []
        license_plate_ids = []
        objects_names = index_ade20k['objectnames']
        for i, labels in enumerate(objects_names):
            if 'person' in labels:
                person_ids.append(i)
                print('person label found with indice %d: %s' % (i, labels))
            if 'license plate' in labels:
                license_plate_ids.append(i)
                print('license plate label found with indice %d: %s' % (i, labels))

        return person_ids, license_plate_ids


    def get_no_burred_images_indices(index_ade20k):

        objects_presence = index_ade20k['objectPresence']
        num_objects, num_images = objects_presence.shape
        person_ids, license_plate_ids = get_person_and_licence_plate_ids(index_ade20k)
        num_persons = np.sum(objects_presence[person_ids, :], axis=0)
        print('%d/%d imágenes tienen personas' % (np.sum(num_persons > 0), num_images))
        num_license_plates = np.sum(objects_presence[license_plate_ids, :], axis=0)
        print('%d/%d imágenes tienen matrículas' % (np.sum(num_license_plates > 0), num_images))
        indices = np.logical_and(num_persons == 0, num_license_plates == 0)
        print('%d imágenes no tienen ni personas ni matrículas' % np.sum(indices))

        return np.arange(num_images)[indices]


    indices = get_no_burred_images_indices(index_ade20k)

    with open(kernels_images_list) as f:
        kernels_list = f.readlines()

    #files = files[:100]
    starttime = time.time()

    pool = multiprocessing.Pool(4)
    pool.map(multiprocessing_func, indices)
    pool.close()


    print()
    print('Time taken = {} seconds'.format(time.time() - starttime))
