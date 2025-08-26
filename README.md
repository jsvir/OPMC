
# One-pass Multi-view Clustering for Large-scale Data (OPMC)

An unofficial pytorch-based implementation of the paper: 

> Jiyuan Liu, Xinwang Liu, Yuexiang Yang, Li Liu, Siqi Wang, Weixuan Liang and Jiangyong Shi: One-pass Multi-view Clustering for Large-scale Data, IEEE International Conference on Computer Vision, ICCV, 2021. (Accepted Jul. 2021)



The original MATLAB-based implementation could be found here: https://github.com/liujiyuan13/OPMC-code_release


### Why is it useful?
- Non-deep, parameter-free multiview clustering algorithm
- Comparative results to the recent baselines
- Could be used with CPU only

### How to run?
1. Unzip a dataset from `datasets` directory or prepare your own dataset
2. Run the next command:
````python run.py  --dataset <PATH TO PICKLED DATASET>````

The dataset is assumed to be stored as a dictionary with keys:
- X - dict with key, value for each view (key = name of the view, value = numpy ndarray matrix for samples)
- Y - dict with key, value for each view (key = name of the view, value = numpy ndarray matrix for labels )

### Example datasets:
- **MSRCv1** Consists of 210 scene recognition images belonging to 7 categories Zhao et al. (2020). Each image is described by 5 different types of features.
- **Scene15** Consists of 4,485 scene images belonging to 15 classes Fei-Fei & Perona (2005).
- **RBGD** Kong et al. (2014): Consists of 1, 449 samples of indoor scenes image-text of 13 classes. We follow the version provided in Trosten et al. (2021a); Zhou & Shen (2020), where image features are extracted from a ResNet50 model pretrained on the ImageNet dataset and text features from a doc2vec model pretrained on the Wikipedia dataset


## 📖 Citation

If you find this repository useful, please cite it:

```bibtex
@software{Svirsky_OPMC_2024,
author = {Svirsky, Jonathan},
month = nov,
title = {{OPMC}},
url = {https://github.com/jsvir/OPMC},
version = {1.0.0},
year = {2024}
}
