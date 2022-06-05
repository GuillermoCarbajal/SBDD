# Rethinking Motion Deblurring Training: A Segmentation-Based Method for Simulating Non-Uniform Motion Blurred Images

## Trained models

[SRN](https://github.com/jiangsutx/SRN-Deblur) and [DeepDeblur](https://github.com/SeungjunNah/DeepDeblur-PyTorch) models trained with SBDD can be downloaded from [here](https://iie.fing.edu.uy/~carbajal/SBDD/trained_models.zip)

## SBDD Dataset 

The dataset used to train the models (with gamma correction) can be downloaded from [here](https://iie.fing.edu.uy/~carbajal/SBDD/SBDD_gamma.zip)

## Dataset Generation

To generate a dataset with the proposed methodology you can follow the following steps:

### 1. Conda environment and requirements

conda create -n SBDD python=3.8
pip install -r dataset/requirements.txt

### 2.COCO, ADE20K and kernels datasets

Download COCO, ADE20K and the kernels used to generate the dataset:

[COCO (2017 train images)](https://cocodataset.org/#download)       
[ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)                
[kernels_SBDD](https://iie.fing.edu.uy/~carbajal/SBDD/kernelsSBDD.zip)

### 3. Generation

Replace the following lines in the script dataset/generate_dataset.sh by your COCO, ADE20K and kernels folder.

ADE_DIR='/media/carbajal/OS/data/datasets/ADE20K_new/carbajal_776b3c5f'     
ADE_INDEX='/media/carbajal/OS/data/datasets/ADE20K_new/carbajal_776b3c5f/ADE20K_2021_17_01/index_ade20k.pkl'      
COCO_DIR='/media/carbajal/OS/data/datasets/COCO/images/train2017'       
KERNELS_DIR='/media/carbajal/OS/data/datasets/kernel_dataset'        

Then, run:

```
bash dataset/generate_dataset.sh
```

## Models Evaluation


When ground-truth blurry-sharp pairs are available (GoPro, DVD, REDS, RealBlur) we used the following code to quantify the restoration quality. We adapted the evaluation code from [RealBlur](https://github.com/rimchang/RealBlur)  repository.

```
cd evaluation
python evaluation_parallel_ecc.py -b datasets/RealBlur/test/blur -s datasets/RealBlur/test/gt -r resultsRealBlur/RealBlur_with_SBDD```

For Kohler dataset, we used:
```
python evaluation_Kohler_parallel_ecc.py -b datasets/KohlerDataset/BlurryImages -s datasets/KohlerDataset/GroundTruthImg -r results_Kohler/Kohler_with_SBDD
```
