# MKE-GCN
This repo is the official implementation for ACFL.


# Data Preparation


#### NTU RGB+D 60 and 120

1. Request dataset here: http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`


### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

# Training & Testing

### Training

- Change the config file depending on what you want.

```
# Example: training MKE-GCN on NTU RGB+D 120 cross subject with GPU 0 1 2 3
python train_net.py --config config/nturgbd120-cross-subject/acfl_ctr_gcn.yaml --work-dir work_dir/ntu120/csub/acfl_ctr_gcn --device 0 1 2 3
```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python test_net.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --weights <work_dir>/xxx.pt --device 0
```

