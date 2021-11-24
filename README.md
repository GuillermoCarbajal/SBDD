# Rethinking Motion Deblurring Training: A Segmentation-Based Method for Simulating Non-Uniform Motion Blurred Images
# CVPR 2022 Submission 11242

## Trained models

Models can be anonymously downloaded from [here](https://drive.google.com/file/d/1Dg7UnSz2ZQmJ4jy0hucQ9od7ldsibwDN/view?usp=sharing)

## Dataset Generation

### 1. Conda environment and requirements

conda create -n SBDD python=3.8
pip install -r dataset/requirements.txt

### 2. ADE20K and kernels datasets

Download ADE20K and the kernels used to generate the dataset:

[ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
[kernels_SBDD]()

### 3. Generation

Replace the following lines in the script dataset/generate_dataset.sh by your ADE20K and kernels folder.

ADE_DIR='/media/carbajal/OS/data/datasets/ADE20K/ADE20K_2016_07_26/images'
KERNELS_DIR='/media/carbajal/OS/data/datasets/kernel_dataset'

Then, run:

```
bash dataset/generate_dataset.sh
```

## Models Evaluation

cd evaluation

```
python evaluation_parallel_ecc.py -b datasets/RealBlur/test/blur -s datasets/RealBlur/test/gt -r /results_deblurring/resultsRealBlur/SRN_trained_with_ADE/RealBlur_ade_ade_sat_483900
```

```
python evaluation_Kohler_parallel_ecc.py -b /media/carbajal/OS/data/datasets/KohlerDataset/BlurryImages -s /media/carbajal/OS/data/datasets/KohlerDataset/GroundTruthImg -r /media/carbajal/OS/data/results_deblurring/results_Kohler/cvpr2022/Kohler_ade_ade_sat_min_400_gf1_483900
```
