# Reperio-rPPG: Relational Temporal Graph Neural Networks for Periodicity Learning in Remote Heart Rate Measurement
This repository is the official implementation of the paper [**Reperio-rPPG: Relational Temporal Graph Neural Networks for Periodicity Learning in Remote Heart Rate Measurement**].

## Dependencies
```
conda create -n rppg python=3.9
conda activate rppg
pip install -r requirements.txt
gdown --id 1TqE3a7VAVLj0d31YKII_PSerdt2aixBP torch-2.0.0+cu118-cp39-cp39-linux_x86_64
gdown --id 1S6rjIxDfawnHROX6EBVD4vLIloMVD2Fc torchvision-0.15.1+cu118-cp39-cp39-linux_x86_64

pip install path/to/torch-2.0.0+cu118-cp39-cp39-linux_x86_64
pip install path/to/torchvision-0.15.1+cu118-cp39-cp39-linux_x86_64
```

## Train & Validation

1. Replace **Path/to/XXXX/dataset** and **Path/to/cache/directory** with the correct directories in *preprocess.py*
2. Execute the script by running: `python preprocess.py`
3. Update the configuration files in *./configs/* by setting **Path/to/XXXX/dataset** and **Path/to/cache/directory** to the appropriate paths.
4. Start training by executing: `python ./train.py --config ./configs/reperio_xxxx.yaml --split_idx idx` where `xxxx` refers to the dataset name and `idx` is the data split index (from 0 to 4).

