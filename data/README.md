# Brain Tumor Classification Dataset

This directory is intended to store the brain tumor MRI dataset files. The dataset consists of multiple ZIP files containing `.mat` files with MRI scans.

## Dataset Structure

The dataset should be organized as follows:
- Multiple `.zip` files named `brainTumorDataPublic_*.zip`
- Each ZIP containing `.mat` files with MRI scans
- Each `.mat` file structured with:
  - `cjdata/image`: The MRI scan data
  - `cjdata/label`: The tumor type label (1-indexed: 1 for meningioma, 2 for glioma, 3 for pituitary tumor)

## Obtaining the Dataset

The dataset can be obtained from:
1. [Brain Tumor Classification Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
2. [Kaggle Brain Tumor Classification Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

## Usage

1. Download the dataset files
2. Place the ZIP files in this directory
3. The training script will automatically handle the extraction and processing of the data

## Data Format

Each `.mat` file contains:
- `cjdata.image`: The MRI scan (grayscale image)
- `cjdata.label`: Tumor type (1: meningioma, 2: glioma, 3: pituitary tumor)
- `cjdata.PID`: Patient ID
- `cjdata.tumorBorder`: Tumor border coordinates
- `cjdata.tumorMask`: Binary mask of the tumor region

## Preprocessing

The data is automatically preprocessed by the training script:
- Images are resized to 384x384 pixels
- Normalized to [0, 1] range
- Augmented with random rotations, flips, and affine transformations during training

## Citation

If you use this dataset in your research, please cite:
```
@article{cheng2017enhanced,
  title={Enhanced performance of brain tumor classification via tumor region augmentation and partition},
  author={Cheng, Jun and Yang, Wei and Huang, Minghuang and Huang, Wei and Jiang, Jing and Zhou, Yu and Yang, Rui and Zhao, Jie and Feng, Yanqiu and Feng, Qianjin and others},
  journal={PloS one},
  volume={10},
  number={10},
  pages={e0140381},
  year={2017},
  publisher={Public Library of Science San Francisco, CA USA}
}
``` 