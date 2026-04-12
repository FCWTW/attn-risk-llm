# Driver Attention Module
Modeified from：https://github.com/ykotseruba/SCOUT

## Deployment Details

1. Download BDDA and DR(eye)VE dataset following the instructions on their official websites:
[BDDA](http://bdd-data.berkeley.edu/download.html)、
[DR(eye)VE](https://aimagelab-legacy.ing.unimore.it/imagelab/page.asp?IdPage=8)

2. Define the following paths in ~/.bashrc：
```bash
# Paths to the datasets and DAM code
export DREYEVE_PATH="/home/wayne/Documents/DREYEVE_DATA"
export BDDA_PATH="/home/wayne/Documents/BDDA"
export CODE_FOLDER="/home/wayne/Documents/Progress/SCOUT"
```
3. The required packages are listed in the environment.yaml file. The model was originally trained using “python=3.8, pytorch=2.0.0, cuda11.8, cudnn8.7.0”.

4. The weights pre-trained on BDDA can be found [here]().

---
## Training the model
Once you have completed the Deployment Details and made the necessary changes to /config/DAM, you can run the following command:
```bash
python3 train.py
```

---
## Testing the model
Place the config file and model weights in /your_config, then run the following command:
```bash
python3 test.py --config_dir /your_config --evaluate
```

If you need to save test images, run the following command:
```bash
python3 test.py --config_dir /your_config --evaluate --save_images
```

---
## Processing the MMAU dataset
```bash
```