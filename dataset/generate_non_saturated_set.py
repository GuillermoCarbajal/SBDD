import os
from skimage.io import imread, imsave
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import fftconvolve
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.util import random_noise
from scipy import io
import argparse
import time
import multiprocessing
from pycocotools.coco import COCO


def get_segmentation_mask(sharp_image, img_id):
    '''
    :param image_fullname:
    :param sharp_image:
    :return:
        segmentation mask:
    '''

    W = sharp_image.shape[1]
    H = sharp_image.shape[0]
    K = kernel_size

    only_background = True

    annIds = coco.getAnnIds(imgIds=img_id, areaRng=[min_objetc_area, np.inf], iscrowd=False)
    anns = coco.loadAnns(annIds)
    n_objects = len(anns)
    # print('Number of objects in image: ', len(anns))
    areas = []
    for n in range(n_objects):
        bbox = anns[n]['bbox']
        p0_x = bbox[0]
        p0_y = bbox[1]
        p1_x = p0_x + bbox[2]
        p1_y = p0_y + bbox[3]
        area = np.abs(p0_x-p1_x) * np.abs(p0_y-p1_y)
        areas.append(area)


    orden = np.argsort(np.asarray(areas))
    num_objects = np.min([len(orden), max_objects_to_blur])
    # print('%d objects added in image %d' % (num_objects, img_id))
    obj_masks = []
    selected_indices = np.asarray(np.arange(len(orden)))[orden[:num_objects]]
    for ind in selected_indices:
        if type(anns[ind]['segmentation']) == list:
            # polygon
            seg = anns[ind]['segmentation'][0]
            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
            mask_to_draw = Image.new('L', (W, H), 0)
            # print(poly.shape)
            poly_to_draw = np.round(poly.flatten()).astype(np.int)
            # print(poly_to_draw, poly_to_draw.dtype,poly_to_draw.shape)
            ImageDraw.Draw(mask_to_draw).polygon(list(poly_to_draw), outline=1, fill=1)
            mask_k = np.array(mask_to_draw)
            # print(mask.shape, mask.dtype, np.unique(mask), np.sum(mask)/(mask.shape[0] * mask.shape[1]))
            if mask_k.sum()>min_objetc_area:
                obj_masks.append(mask_k)
            # plt.figure()
            # plt.imshow(128 * mask_k, cmap='gray')
            # plt.show()
        else:
            print('BE CAREFUL: segmentation key does not exit, mask cannot be created for %d,  indice %d' % (img_id, ind))
    # else:
    #     # mask
    #     # t = img[ann['image_id']]
    #     print(anns[ind]['segmentation'])
    #     if type(anns[ind]['segmentation']['counts']) == list:
    #         rle = maskUtils.frPyObjects([anns[ind]['segmentation']], H, W)
    #     else:
    #         rle = [anns[ind]['segmentation']]
    #     m = maskUtils.decode(rle)
    #     # m_img = np.ones( (m.shape[0], m.shape[1], 3) )
    #     # if ann['iscrowd'] == 1:
    #     #    color_mask = np.array([2.0,166.0,101.0])/255
    #     # if ann['iscrowd'] == 0:
    #     #    color_mask = np.random.random((1, 3)).tolist()[0]
    #     # for i in range(3):
    #     #    m_img[:,:,i] = color_mask[i]
    #
    #     if anns[ind]['iscrowd'] == 1:
    #         print('is crowd')
    #     plt.figure()
    #     # plt.imshow(np.dstack( (m_img, m*0.5)), cmap='gray')
    #     plt.imshow(128 * m, cmap='gray')
    #     plt.show()

    if len(obj_masks) > max_objects_to_blur:
        print('Ojo')

    return obj_masks

def generate_blurry_sharp_pair(sharp_image_crop, kernels, masks_to_send, kernel_size=33, gray_scale=False,
                               gamma_correction=True, gamma_factor=2.2, augment_illumination=True,
                               jitter_illumination=1.0, poisson_noise=False, gaussian_noise=False, noise_std=0.01):

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

def process(image_id):


    image_data = coco.loadImgs([image_id])[0]
    file_name = image_data['file_name']
    sharp_image = np.array(Image.open(os.path.join(COCO_DIR, file_name)))
    img_name, ext = file_name.split('.')

    segmentation_masks = get_segmentation_mask(sharp_image, image_id)

    if len(sharp_image.shape) == 3:
        M, N, C = sharp_image.shape
    else:
        M, N = sharp_image.shape
        C = 1
    min_size = np.min([M, N])


    if C > 1 and (min_size > min_img_size):
        # An interesting crop is chosen
        #B, values_of_segmented_instances_in_crop, values_of_segmented_instances, corrected_raw_names = get_segmentation_info(
        #    image_fullname)

        for n in range(n_images):

            blurry_image_filename = os.path.join(output_dir, 'blurry', img_name + '_%d.jpg' % n)
            sharp_image_filename = os.path.join(output_dir, 'sharp', img_name + '_%d.jpg' % n)

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

                for object_mask in segmentation_masks:

                    idx_kernel = np.random.randint(len(kernels_list))
                    kernel_name = kernels_list[idx_kernel][0:-1].split('/')
                    kernel = io.loadmat(os.path.join(Kernels_Dir, kernel_name[1]))['K']

                    kernels.append(kernel)
                    masks.append(object_mask.astype(np.float32))
                    masks[0][object_mask == 1] = 0  # se ponen a cero las posiciones del background que fueron sustituidas

                for k, m in zip(kernels, masks):
                    print('image_id = %d; n=%d; img shape = %s; num pixels kernel = %f; num pixels masks = %f ' % (image_id, n, sharp_image.shape, (k>0).sum(), (m==1).sum()))

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
    #n, file_name = n_file_name
    #print('{}: filename is {}'.format(n, file_name ))
    process(n_file_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--kernels_dir', '-kd', type=str, help='kernels root folder', required=False,
<<<<<<< HEAD
                        default='/media/carbajal/OS/data/datasets/kernel_dataset/size_33_exp_1/train')
    parser.add_argument('--coco_root_dir', '-ad', type=str, help='COCO dir folder', required=False,
                        default='/media/carbajal/OS/data/datasets/COCO/images/train2017')
=======
                        default='data/datasets/kernel_dataset/size_33_exp_1')
    parser.add_argument('--ade_dir', '-ad', type=str, help='ADE dir folder', required=False,
                        default='data/datasets/ADE20K/ADE20K_2016_07_26/images')
>>>>>>> 233bf116193c689ea198a0f0369125ba6be872c7
    parser.add_argument('--kernels_list', '-kl', type=str, help='kernels list', required=False,
                        default='blur_kernels_train.txt')
    parser.add_argument('--coco_json_path', '-al', type=str, help='json file', required=False,
                        default='annotations/instances_train2017.json')
    parser.add_argument('--K', type=int, help='kernel size', required=False, default=33)
    parser.add_argument('--min_img_size', type=int, help='min crop size', required=False, default=400)
    parser.add_argument('--min_object_area', type=int, help='min object area', required=False, default=4000)
    parser.add_argument('--max_objects_to_blur', type=int, help='max objects to blur', required=False, default=5)
    parser.add_argument('--output_dir', '-o', type=str, help='path of the output dir', required=True)
    parser.add_argument('--gamma_factor', '-gf', type=float, default=2.2, help='gamma_factor', required=False)
    parser.add_argument('--augment_illumination', default=True, action='store_true',
                        help='whether to augment illumination')
    parser.add_argument('--n_images', type=int, help='number of blurry images per sharp image', required=False, default=2)

    opt = parser.parse_args()
    COCO_DIR = opt.coco_root_dir
    Kernels_Dir = opt.kernels_dir
    json_file_path = opt.coco_json_path
    kernels_images_list = opt.kernels_list
    output_dir = opt.output_dir
    min_img_size = opt.min_img_size
    kernel_size = opt.K
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


    supercategories = ['person', 'vehicle', 'animal']
    coco = COCO(os.path.join(COCO_DIR, '../..' , json_file_path))
    labels_to_blur = coco.getCatIds(supNms=supercategories)


    images_of_interest = []
    keys = coco.anns.keys()
    for key in keys:
        min_area_cond = coco.anns[key]['area'] > min_objetc_area
        is_crowd = coco.anns[key]['iscrowd']
        cat_cond = coco.anns[key]['category_id'] in labels_to_blur
        if cat_cond and min_area_cond and not is_crowd:
            images_of_interest.append(coco.anns[key]['image_id'])

    training_files_list = np.unique(np.array(images_of_interest))
    print('Number of training images: %d' % len(training_files_list))

    with open(kernels_images_list) as f:
        kernels_list = f.readlines()

    # files = files[:100]
    starttime = time.time()

    pool = multiprocessing.Pool(4)
    pool.map(multiprocessing_func, training_files_list)
    pool.close()


    print()
    print('Time taken = {} seconds'.format(time.time() - starttime))
