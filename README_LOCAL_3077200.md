# GCN-DevLSTM: Path Development for Skeleton-Based Action Recognition
![Alt text](figs/framework.png?raw=true "Title")
This repository is the official implementation of the paper entitled "GCN-DevLSTM: Path Development for Skeleton-Based Action Recognition". 

## Datasets
We provide configurations for three datasets:

-NTU RGB+D 60 skeleton
-NTU RGB+D 120 skeleton
-Chalearn 2013 skeleton

## Requirements

* numpy
* torch
* tqdm

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - chalearn/
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
    - NTU_RGBD_samples_with_missing_skeletons.txt
    - NTU_RGBD120_samples_with_missing_skeletons.txt
```

#### Generating Data

1. NTU RGB+D 60 or 120
    - `cd data/ntu or data/ntu120`
    - `python get_raw_skes_data.py`
    - `python get_raw_denoised_data.py`
    - `python seq_transformation.py`

## Training & Testing

- To train a new GCN-DevLSTM model run:
```
./train.sh
```

- To train model on NTU RGB+D 60/120 with bone, motion or dual graph modalities, setting bone/vel/labeling_mode arguments in the config file ntu_sub/train_joint.yaml.
```
set 'bone: False and vel: False' # use joint modality
set 'bone: True and vel: False' # use bone modality
set 'bone: False and vel:True' # use joint motion modality
set 'bone: True and vel: True' # use bone motion modality
set 'bone: True and vel: False and labeling_mode: dual_graph'  # use dual graph modality
```


- To test a trained model:
```
./test_NTU.sh
```

- To ensemble the results of different modalities, run the following command:
```
./ensemble.sh
```



- Examples
  - Train on NTU 120 XSub Joint on device 0
    - `python main.py --config ./config/ntu_sub/train_joint.yaml --device 0`
  - Train on Chalearn 2013
    - `python main.py --config ./config/chalearn/train_joint.yaml --device 0`
  - The model used is in `model/gcn_devLSTM.py`

## Acknowledgements

We want to thank the authors of the following papers and repositories, their work formed the basis for this repository
  - [GCN-LogsigRNN](https://github.com/steveliao93/GCN_LogsigRNN/tree/main)
  - [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN/tree/main)
  - [DevNet](https://github.com/PDevNet/DevNet)
  - [Hyperformer](https://github.com/ZhouYuxuanYX/Hyperformer/tree/main)




  
