ADE_DIR='/media/carbajal/OS/data/datasets/ADE20K_new/carbajal_776b3c5f'
ADE_INDEX='/media/carbajal/OS/data/datasets/ADE20K_new/carbajal_776b3c5f/ADE20K_2021_17_01/index_ade20k.pkl'
COCO_DIR='/media/carbajal/OS/data/datasets/COCO/images/train2017'
KERNELS_DIR='/media/carbajal/OS/data/datasets/kernel_dataset'

unzip $KERNELS_DIR/kernelsSBDD.zip -d $KERNELS_DIR
unzip $KERNELS_DIR/kernels_size33.zip -d $KERNELS_DIR/kernels_size33
unzip $KERNELS_DIR/kernels_size65.zip -d $KERNELS_DIR/kernels_size65

python generate_non_saturated_set.py --coco_root_dir $COCO_DIR --kernels_dir $KERNELS_DIR/kernels_size65/train --max_objects_to_blur 2 --K 65 --gamma_factor 2.2 --augment_illumination --n_images 1 --min_img_size 400 --output_dir $OUTPUT_DIR/COCO_ks65_aug_mob2 

python generate_saturated_set.py --ade_dir $ADE_DIR --ade_index $ADE_INDEX --kernels_dir $KERNELS_DIR/kernels_size33/train --gamma_factor 2.2 --augment_illumination --n_images 8 --min_img_size 400 --output_dir $OUTPUT_DIR/ADENew_Sat_ks33_aug

python generate_saturated_set.py --ade_dir $ADE_DIR --ade_index $ADE_INDEX --kernels_dir $KERNELS_DIR/kernels_size65/train --K 65 --gamma_factor 2.2 --augment_illumination --n_images 8 --min_img_size 400 --output_dir $OUTPUT_DIR/ADENew_Sat_ks65_aug 

python generate_training_lists.py --dataset_dir $OUTPUT_DIR
