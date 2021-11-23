# Path to ADE dataset images folder,  ADE_DIR=path_to/ADE20K_2016_07_26/images
ADE_DIR='/media/carbajal/OS/data/datasets/ADE20K/ADE20K_2016_07_26/images'
KERNELS_DIR='/media/carbajal/OS/data/datasets/kernel_dataset'
OUTPUT_DIR='SBDD_gamma'

unzip $KERNELS_DIR/kernelsSBDD.zip -d $KERNELS_DIR
unzip $KERNELS_DIR/kernels_size33.zip -d $KERNELS_DIR/kernels_size33
unzip $KERNELS_DIR/kernels_size65.zip -d $KERNELS_DIR/kernels_size65

python generate_non_saturated_set.py --ade_dir $ADE_DIR --kernels_dir $KERNELS_DIR/kernels_size33/train --gamma_factor 2.2 --augment_illumination --n_images 1 --min_img_size 400 --output_dir $OUTPUT_DIR/ADE_ks33_aug

python generate_non_saturated_set.py --ade_dir $ADE_DIR --kernels_dir $KERNELS_DIR/kernels_size65/train --K 65 --gamma_factor 2.2 --augment_illumination --n_images 1 --min_img_size 400 --output_dir $OUTPUT_DIR/ADE_ks65_aug 

python generate_saturated_set.py --ade_dir $ADE_DIR --kernels_dir $KERNELS_DIR/kernels_size33/train --gamma_factor 2.2 --augment_illumination --n_images 4 --min_img_size 400 --output_dir $OUTPUT_DIR/ADE_Sat_ks33_aug

python generate_saturated_set.py --ade_dir $ADE_DIR --kernels_dir $KERNELS_DIR/kernels_size65/train --K 65 --gamma_factor 2.2 --augment_illumination --n_images 4 --min_img_size 400 --output_dir $OUTPUT_DIR/ADE_Sat_ks65_aug 

python generate_training_lists.py --dataset_dir $OUTPUT_DIR
