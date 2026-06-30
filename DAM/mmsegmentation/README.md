```bash
conda create --name mmseg python=3.8
conda activate mmseg
conda install pytorch=2.1.2 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install fsspec

pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install "mmsegmentation>=1.0.0"

pip install ftfy regex
```

---
## Code usage
Clone the [official GitHub code](https://github.com/open-mmlab/mmsegmentation/tree/main) and modify the following code:

### check.py
* Place it in『/mmsegmentation』
* Provided by the official source to verify that the environment is properly set up

### inference.py
* Place it in『/mmsegmentation/mmseg/apis』
* The color palette (show_result_pyplot) has been modified to turn irrelevant objects black.

### process_BDDA.py
* Place it in『/mmsegmentation』
* Also place the model weights (.pth) in『/mmsegmentation』.
* Originally, we used『mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024』.
* The config uses the version originally provided by the code; please remember to change the path.
* Remember to adjust the path where BDDA is located, otherwise the code will not execute.

### process_MMAU.py
* Place it in『/mmsegmentation』
* Remember to adjust the path and architecture of MM-AU dataset, otherwise the code will not execute:
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

---
## Reference:
1. https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation
2. https://mmcv.readthedocs.io/en/latest/get_started/installation.html