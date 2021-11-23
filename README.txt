## Dataset Generation

### Conda environment and requirements

conda create -n SBDD python=3.8
pip install -r dataset/requirements.txt

### ADE20K and kernels

Download ADE20K and the kernels used to generate the dataset:

[ADE20K]()
[kernels_SBDD]()

### Generation

Replace the following lines in the script dataset/generate_dataset.sh by your ADE20K and kernels folder.

ADE_DIR='/media/carbajal/OS/data/datasets/ADE20K/ADE20K_2016_07_26/images'
KERNELS_DIR='/media/carbajal/OS/data/datasets/kernel_dataset'

Then, run:

```
bash dataset/generate_dataset.sh
```

