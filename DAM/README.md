# Driver Attention Module

## Deployment details for Driver Attention Module

1. Define the following paths in ~/.bashrc：
```bash
# Paths to the datasets and DAM code
export DREYEVE_PATH="/home/wayne/Documents/DREYEVE_DATA"
export BDDA_PATH="/home/wayne/Documents/BDDA"
export EXTRA_ANNOT_PATH="/home/wayne/Documents/Progress/SCOUT/extra_annotations"
export CODE_FOLDER="/home/wayne/Documents/Progress/SCOUT"
```
2. The required packages are listed in the environment.yaml file. The model was originally trained on an NVIDIA GeForce RTX 4090 using Python 3.8, PyTorch 2.0.0, CUDA 11.8, and cuDNN 8.7.0.

3. The weights pre-trained on BDDA can be found [here](https://drive.google.com/file/d/1RCpoOry9epnHwKJ6pTBhQxXmKaE5LUkG/view?usp=drive_link).

4. You can modify the model type, dataset type, and training parameters in [/config/DAM.yaml](/DAM/config/DAM.yaml).

5. Before training or testing, you need to process the BDDA dataset using [mmsegmentation](/DAM/mmsegmentation/process_BDDA.py).

---
## Inference on MM-AU dataset

```
MM-AU # root of your MM-AU
├── CAP-DATA
│   ├── 1
│       ├── 001537
│           ├── images
│               ├── 000001.jpg
│               ├── ......
│   ├── 2
│   ├── ......
│   ├── 62
├── DADA-DATA
│   ├── 1
│       ├── 001
│           ├── images
│               ├── 0001.png
│               ├── ......
│   ├── 2
│   ├── ......
│   ├── 61
```
1. Please organize the dataset according to the above structure.

2. Next, use [/mmsegmentation/process_MMAU.py](/DAM//mmsegmentation/process_MMAU.py) to generate a semantic segmentation map of the dataset.

3. Eventually, use [MMAU_inference.py](/DAM/scripts/MMAU_inference.py) to perform DAM inference.

---
## Results Showcase

coming soon...