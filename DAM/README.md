# Deployment details for Driver Attention Module

1. Define the following paths in ~/.bashrc：
```bash
# Paths to the datasets and DAM code
export DREYEVE_PATH="/home/wayne/Documents/DREYEVE_DATA"
export BDDA_PATH="/home/wayne/Documents/BDDA"
export CODE_FOLDER="/home/wayne/Documents/Progress/SCOUT"
```
3. The required packages are listed in the environment.yaml file. The model was originally trained using “python=3.8, pytorch=2.0.0, cuda11.8, cudnn8.7.0”.

4. The weights pre-trained on BDDA can be found [here]().

5. You can modify the model type, dataset type, and training parameters in [/config/DAM.yaml]().

6. Before training or testing, you need to process the BDDA dataset using [mmsegmentation]().

---
## Prepare for MM-AU dataset

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

2. Next, use [/mmsegmentation/process_MMAU.py]() to generate a semantic segmentation map of the dataset.

3. Eventually, use [MMAU_inference.py]() to perform DAM inference.