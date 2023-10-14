GOPRO_LARGE_FOLDER='/media/carbajal/OS/data/datasets/GOPRO_Large/original-data'
SEGMENTATION_INFO_FOLDER='/media/carbajal/OS/data/datasets/GoPro_detectron2_segmentation'
KERNELS_FOLDER='/media/carbajal/OS/data/datasets/kernel_dataset/ks65_texp05_F1000/train'

PREPARED_DATA_FOLDER='prepared-data'
GAMMA_FACTOR=2.2
AUGMENT_ILLUMINATION=True
MOVING_OBJECTS=5
OUTPUT_FOLDER='SBDD_NU_mob5_gamma22_ill_aug_2up'


python prepare_data.py --GOPRO_original_data_folder ${GOPRO_LARGE_FOLDER} --prepared_data_folder ${PREPARED_DATA_FOLDER}


# sample dataset with multiplicative augmentation and gamma correction factor 2.2
python generate_GoPro_dataset.py  --kernels_dir ${KERNELS_FOLDER} --gopro_sharp_dir ${PREPARED_DATA_FOLDER}/sharp/train --gopro_segmentation_dir ${SEGMENTATION_INFO_FOLDER} --kernels_list 'blur_kernels_train.txt' --gamma_factor 2.2 --augment_illumination --output_dir=${OUTPUT_FOLDER}
