# A Driving Risk Prediction Framework Integrating Driver Attention and Large Language Models

![overview](/image/Overview.png)

## Setting up datasets
* Download BDDA dataset following the instructions on [official websites](http://bdd-data.berkeley.edu/download.html).
* Download DR(eye)VE dataset following the instructions on [official websites](https://aimagelab-legacy.ing.unimore.it/imagelab/page.asp?IdPage=8).
* Download TrafficGaze dataset following the instructions on [huggingface](https://huggingface.co/datasets/springyu/TrafficGaze).
* Download MM-AU dataset following the instructions on [huggingface](https://huggingface.co/datasets/JeffreyChou/MM-AU/tree/main).

---
## Driver Attention Module (DAM)

![DAM](/image/DAM.png)

Modified from [SCOUT+](https://github.com/ykotseruba/SCOUT). Those code removes certain parts of the training and inference code for SCOUT and SCOUT+. If you need these files, please refer to the original GitHub repository.

### Environment Setup
You can find instructions on how to set up the environment [here](/DAM/README.md).

### Training the model
Once you have completed the Deployment Details and made the necessary changes to [/config/DAM.yaml](/DAM/config/DAM.yaml), you can run the following command for training:
```bash
python3 train.py
```

Or, if you want to use a customized config.yaml file for training, run the following command:
```bash
python3 train.py --config_dir /your_config.yaml
```

### Testing the model
Place the config file and model weights in /your_config, then run the following command for:
```bash
python3 test.py --config_dir /your_config --evaluate
```

If you need to save test images, run the following command:
```bash
python3 test.py --config_dir /your_config --evaluate --save_images
```

### Visualize the results
You can use [get_heatmap.py](/DAM/scripts/get_heatmap.py) to get the visual heatmap of driver attention.
```bash
python3 get_heatmap.py --gazemap_dir /gazemap --rgb_dir /camera_frames --output_dir /your_output
```

### Inference on MM-AU dataset
You must first follow the [steps](/DAM/README.md) before executing the following command:
```bash
python3 MMAU_inference.py --config_dir /your_config --dataset_dir /MM-AU_root
```

---
## Risk Assessment Module (RAM)

![RAM](/image/RAM.png)

Modified from：https://github.com/DeSinister/CycleCrash/

You can find instructions on how to set up the environment [here](/RAM/README.md).

---
## LLM Inference Module (LIM)

![LIM](/image/LIM.png)

You can find instructions on how to set up the environment [here](/LIM/README.md).