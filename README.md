Code associated to the ICIP 2024 submission:

# Improving Generalization of Deep Motion Deblurring Networks: A Convolution-Based Procedure for Analyzing and Addressing the Limitations of Current Benchmark Datasets
G Carbajal, P Vitoria, P Musé, J Lezama


> Successfully training end-to-end deep networks for real motion deblurring requires datasets of sharp/blurred image pairs that are realistic and diverse enough to achieve generalization to real blurred images. Obtaining such datasets remains a challenging task. In this paper, we first review the limitations of existing deblurring benchmark datasets from the perspective of generalization to blurry images in the wild and analyze the underlying causes. Based on this analysis, we propose an efficient procedural methodology to generate sharp/blurred image pairs based on a simple yet effective model for forming blurred images. This allows for generating virtually unlimited realistic and diverse training pairs.  
We demonstrate the effectiveness of the proposed dataset by training existing deblurring architectures on the simulated pairs and performing cross-dataset evaluation on synthetic and real blurry images. When training with the proposed method, we observed superior generalization performance for deblurring real motion-blurred photos of dynamic scenes.

State-of-the-art deblurring neural networks achieve spectacular restorations in the GoPro dataset, but generalize poorly to real non-uniformly blurred images (as shown in the figure below).            
<img src="figs/motivation.png"  height=600 width=1200 alt="SRN Results when trained with SBDD  ">   

## Trained models

We provide the [SRN](https://github.com/jiangsutx/SRN-Deblur) models trained with the proposed Segmentation Based Deblurring Dataset (SBDD). We also provide links to models trained with other datasets to facilitate the comparison.   

| Arch \ Dataset |   GoPro |  REDS  |  SBDD  |          
|-------|:---------------------|:--------------------|---------------------|         
| SRN   | [public model](https://iie.fing.edu.uy/~carbajal/SBDD_data/SBDD_models/srn-models/GoPro_color.zip) | [trained](https://iie.fing.edu.uy/~carbajal/SBDD_data/SBDD_models/srn-models/REDS_color.zip) | [trained](https://iie.fing.edu.uy/~carbajal/SBDD_data/SBDD_models/srn-models/SRN_SBDD_models.zip) |    
    

## Testing SRN Models

### 1. Conda environment and requirements
```
conda create -n srn-py27 python=2.7
conda activate srn-py27
pip install scipy scikit-image numpy tensorflow-gpu==1.12
conda install cudatoolkit==9.0 cudnn==7.6.5
```

### 2. Test the model

```
cd SRN-Model
python run_model.py --input_path=../sample_images --output_path=../sample_results --training_dir model_folder --step iteration_number

# Example
# python run_model.py --input_path=../sample_images --output_path=../sample_results --training_dir  /media/carbajal/OS/data/models/srn_models/SRN_SBDD_models/GoPro_uniform_ks65_texp05_F1000_ill_aug_2up_n10_ef5 --step 262800
```

<img src="figs/srn.png"  height=575 width=1200 alt="SRN Results when trained with SBDD  ">   

## Dataset Generation

To generate a dataset with the proposed methodology you can follow the following steps:

### 1. Conda environment and requirements

conda create -n SBDD python=3.8     
pip install -r requirements.txt    

### 2.GoPro dataset, segmentation masks and kernels datasets

Download the GoPro dataset, the segmentation masks, and the kernels used to generate the dataset:

[GOPRO_LARGE_FOLDER](https://seungjunnah.github.io/Datasets/gopro.html)                
[SEGMENTATION_INFO_FOLDER](https://iie.fing.edu.uy/~carbajal/SBDD_data/GoPro_detectron2_segmentation.zip)      
[kernels_SBDD](https://iie.fing.edu.uy/~carbajal/SBDD_data/ks65_texp05_F1000_kernels.zip)

### 3. Generation

Unzip the previous file and then replace the following lines in the script dataset/generate_dataset.sh by your *<GOPRO_LARGE_FOLDER>*, *<SEGMENTATION_INFO_FOLDER>* and *<KERNELS_FOLDER>*. 

GOPRO_LARGE_FOLDER='/media/carbajal/OS/data/datasets/GOPRO_Large/original-data'
SEGMENTATION_INFO_FOLDER='/media/carbajal/OS/data/datasets/GoPro_detectron2_segmentation'     
KERNELS_FOLDER='/media/carbajal/OS/data/datasets/kernel_dataset/ks65_texp05_F1000/train'        

Then, run:

```
cd datasets
bash generate_dataset.sh
```

## Sample Dataset 

An instance of our dataset generation procedure can can be downloaded from [here](https://iie.fing.edu.uy/~carbajal/SBDD_data/SBDD_NU_ill_aug_2up_gf22_n10.zip). It was generated with non uniform blur, asuming a gamma correction factor of 2.2, and multiplicative augmentation. The parameter used to generate de dataset can be found inside the dataset folder. 

## Models Evaluation


When ground-truth blurry-sharp pairs are available (GoPro, DVD, REDS, RealBlur) we used the following code to quantify the restoration quality. We adapted the evaluation code from [RealBlur](https://github.com/rimchang/RealBlur)  repository.

```
cd evaluation
python evaluation_parallel_ecc.py -b datasets/RealBlur/test/blur -s datasets/RealBlur/test/gt -r resultsRealBlur/RealBlur_with_SBDD
```

For Kohler dataset, we used: 

```
python evaluation_Kohler_parallel_ecc.py -b datasets/KohlerDataset/BlurryImages -s datasets/KohlerDataset/GroundTruthImg -r results_Kohler/Kohler_with_SBDD
```


