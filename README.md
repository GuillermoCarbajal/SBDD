# Rethinking Motion Deblurring Training: A Segmentation-Based Method for Simulating Non-Uniform Motion Blurred Images
# CVPR 2022 Submission 11242

## Trained models

[SRN](https://github.com/jiangsutx/SRN-Deblur) and [DeepDeblur](https://github.com/SeungjunNah/DeepDeblur-PyTorch) models trained with SBDD can be anonymously downloaded from [here](https://drive.google.com/file/d/1Dg7UnSz2ZQmJ4jy0hucQ9od7ldsibwDN/view?usp=sharing)

## SBDD Dataset 

The dataset used for train models can be downloaded from [here](https://www.dropbox.com/sh/8befj2azfz9w5rs/AAC_R9IB4Z3MCeFFg2OaPIAfa?dl=0)

## Dataset Generation

To generate a dataset with the proposed methodology you can follow the following steps:

### 1. Conda environment and requirements

conda create -n SBDD python=3.8
pip install -r dataset/requirements.txt

### 2. ADE20K and kernels datasets

Download ADE20K and the kernels used to generate the dataset:

[ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
[kernels_SBDD]()

### 3. Generation

Replace the following lines in the script dataset/generate_dataset.sh by your ADE20K and kernels folder.

ADE_DIR='data/datasets/ADE20K/ADE20K_2016_07_26/images'      
KERNELS_DIR='data/datasets/kernel_dataset'

Then, run:

```
bash dataset/generate_dataset.sh
```

## Models Evaluation



When ground-truth blurry-sharp pairs are available (GoPro, DVD, REDS, RealBlur) we used the following code to quantify the restoration quality. We adapted the evaluation code from [RealBlur](https://github.com/rimchang/RealBlur)  repository.

```
cd evaluation
python evaluation_parallel_ecc.py -b datasets/RealBlur/test/blur -s datasets/RealBlur/test/gt -r /results_deblurring/resultsRealBlur/SRN_trained_with_ADE/RealBlur_ade_ade_sat_483900
```

For Kohler dataset, we used:
```
python evaluation_Kohler_parallel_ecc.py -b /media/carbajal/OS/data/datasets/KohlerDataset/BlurryImages -s /media/carbajal/OS/data/datasets/KohlerDataset/GroundTruthImg -r /media/carbajal/OS/data/results_deblurring/results_Kohler/cvpr2022/Kohler_ade_ade_sat_min_400_gf1_483900
```
