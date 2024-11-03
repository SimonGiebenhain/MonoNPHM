#  MonoNPHM: Dynamic Head Reconstruction from Monoculuar Videos
[**Paper**](https://arxiv.org/abs/2312.06740) | [**Video**](https://www.youtube.com/watch?v=n-wjaC3UIeE) | [**Project Page**](https://simongiebenhain.github.io/MonoNPHM/) <br>

<div style="text-align: center">
<img src="mononphm_teaser.gif" width="600"/>
</div>

This repository contains the official implementation of the paper:

### MonoNPHM: Dynamic Head Reconstruction from Monoculuar Videos
[Simon Giebenhain](https://simongiebenhain.github.io/), 
[Tobias Kirschstein](https://niessnerlab.org/members/tobias_kirschstein/profile.html), 
[Markos Georgopoulos](https://scholar.google.com/citations?user=id7vw0UAAAAJ&hl=en),
[Martin Rünz](https://www.martinruenz.de/), 
[Lourdes Agaptio](https://scholar.google.com/citations?user=IRMX4-4AAAAJ&hl=en) and 
[Matthias Nießner](https://niessnerlab.org/members/matthias_niessner/profile.html)  
**CVPR 2024 Highlight**  





## 1. Installation

### 1.1 Dependencies

a) Setup a conda environment and activate it via

```
conda env create -f environment.yml   
conda activate mononphm
```
which creates a new enivornment named `mononphm`. (Installation may take some time).

b) 
Next, manually install `Pytorch` related packages. MonoNPHM depends on Pytorch3D and PytorchGeometric, which can sometimes be tricky to install.
On Linux the following order of commands worked for us:

```
# Install pytorch with CUDA support
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install PytorchGeometry and helper packages with CUDA support
conda install pyg -c pyg
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Install Pytorch3D with CUDA support
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d=0.7.4 -c pytorch3d
```

Finally, fix the `numpy` version using
```
pip uninstall numpy
pip install numpy==1.23
pip install pyopengl==3.1.5
```

c) Install the `mononphm` pacakge in an editable way by running
```
pip install -e .
```


### 1.2 Environment Paths
All paths to data / models / infernce are defined by environment variables.
For this we recomment to create a file in your home directory in `~/.config/mononphm/.env` with the following content:
```
MONONPHM_CODE_BASE="{LOCATION OF THIS REPOSITORY}"
MONONPHM_TRAINING_SUPERVISION="{LOCATION WHERE TRAINING SUPERVISION DATA WILL BE STORED}"
MONONPHM_DATA="{LOCATION OF NPHM DATASET}"
MONONPHM_EXPERIMENT_DIR="{LOCATION FOR TRAINING RUNS }"
MONONPHM_DATA_TRACKING="{LOCATION FOR TRACKING INPUT}"
MONONPHM_TRACKING_OUTPUT="{LOCATION FOR TRACKING OUTPUT}"
```
Replace the `{...}` with the locations where data / models / experiments should be located on your machine.

If you do not like creating a config file in your home directory, you can instead hard-code the paths in the env.py. 
Note that using the `.config` folder can be great advantage when working with different machines, e.g. a local PC and a GPU cluster.



### 1.3 Installing the Preprocessing Pipeline [Not necessary for running a demo]

Our tracking alogorithm relies on FLAME tracking as initiliazation. Therefore, you will need an account for the [FLAME website](https://flame.is.tue.mpg.de/).

To clone the necessary repositories for preprocessing and perform minor code adjustments thereof, run
```
bash install_preprocessing_pipeline.sh
```

Finally, you will need to download the weights for the employed [PIPNet facial landmark detector](https://github.com/jhb86253817/PIPNet) from [here](https://drive.google.com/drive/folders/17OwDgJUfuc5_ymQ3QruD8pUnh5zHreP2).
Download the folder `snapshots/WFLW` and place it into `src/mononphm/preprocessing/PIPnet/snapshots`. 

Next, download the weights `modnet_webcam_portrait_matting.ckpt` from [here](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR) for [MODNet](https://github.com/ZHKKKe/MODNet/tree/master).
Then place them in `src/mononphm/preprocessing/MODNet/pretrained`.

## 2. Data and Model Checkpoints

### Demo Data

You can download the demo data from [here](https://drive.google.com/drive/folders/1XHHabTt_IgYPmGZj0Gj1dyTm7dwyGEvb?usp=sharing) and move the conentens into the folder specified by `MONONPHM_DATA_TRACKING`. We provide 6 examples from the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) alongside the preprocessing results.

### NPHM dataset

Additionally, we provide an extension to the [NPHM Dataset](https://simongiebenhain.github.io/NPHM/), which now contains 488 people. To download the data, you will need to fill out the [Terms of Service](https://docs.google.com/forms/d/e/1FAIpQLScG9BhoHelqV6GnT-z9P2TsGTJ2x_FPHxdnne_RmlRkbYPPQQ/viewform).

### Model Checkpoints

We provide pretrained models [here](https://drive.google.com/drive/folders/1shwQnL-TBI4vTsKVLOqyQ7B9rQcW9ozW?usp=sharing). Place the contents into `MONONPHM_EXPERIMENT_DIR`. 


### MonoNPHM Test Data

Our test data from the MonoNPHM paper can be downloaded, after agreeing to the Terms of Service [here](https://forms.gle/1EsV5Ezs68N1LYv66).


## 3. Usage

### 3.1 Demo

You can run single-image head reconstruction on a few FFHQ examples using

```
python scripts/inference/rec.py --model_type nphm --exp_name pretrained_mononphm --ckpt 2500 --seq_name FFHQ_ID --no-intrinsics_provided --downsample_factor 0.33 --no-is_video
```
 where `FFHQ_ID` can be one of the folder names of the provided demo data.  
 You will find the results in `MONONPHM_TRACKING_OUTPUT/stage1/FFHQ_ID`.


### 3.2 Running the Preprocessing

When working with your own data, you will need to run the preprocessing pipeline, including landmark detection, facial segmentation, background matting, and FLAME tracking to initialize the head pose.
To this end you can run

```
cd scripts/preprocessing
bash run.sh 510_seq_4 --intrinsics_provided
```
which will run all neccessary steps for the sequence named `510_seq_4` located in `MONONPHM_DATA_TRACKING`. 
The `intrinsics_provided` flag reads the `camera_intrinsics.txt` from the `env_paths.ASSETS` folder and provides the metrical tracker with it.

### 3.3 Tracking

For MonoNPHM tracking, run

```
python scripts/inference/rec.py --model_type nphm --exp_name pretrained_mononphm_original --ckpt 2500 --seq_name 510_seq_4 --intrinsics_provided --is_video
python scripts/inference/rec.py --model_type nphm --exp_name pretrained_mononphm_original --ckpt 2500 --seq_name 510_seq_4 --intrinsics_provided --is_video --is_stage2
```

for the stage 1 and stage 2 of our proposed optimization. (Note that stage2 optimization is only needed for videos.)  
The results can be found in `MONONPHM_TRACKING_OUTPUT/EXP_NAME/stage1/510_seq_4`.

### 3.5 Evaluation

After having tracked the kinect videos, you can run the evaluation scrip using

```
python scripts/evaluation/eval.py --model_name pretrained_mononphm_original
```

There is an `--is_debug` flage that can be used to visualize the individual steps which are necessary to perform, before computing Chamfer-style metrics.
The average the results across all sequences, you can use the `scripts/evaluation/gather_metrics.py`



### 3.4 Training

To train a model yourself, you will first need to generate the necessary training supervision data. Running

```
python scripts/data_processing/compute_fields_new.py --starti START_PID --endi END_PID
python scripts/data_processing/compute_deformation_field.py
```
will create the necessray data. Note that especially `compute_fields_new.py` can take a long time and consumes a lot of storage. (It is possible to reduce the hard-coded number of training samples per scan in `scripts/data_processing/compute_fields_new.py`).  
`START_PID` and `END_PID` specify the range of participant IDs for which the training data will be computed (exlcufing `END_PID`).

To start the training itself, run
```
python scripts/training/launch_training.py --model_type nphm --cfg_file scripts/configs/mononphm.yaml --exp_name  MODEL_NAME --color_branch
```

If you are training on a headless machine, prepending `PYOPENGL_PLATFORM=osmesa` might be necessary.
For our experiments we used 4 GPUs. By detault the training script will use all available GPUs on your machine, and the `batch_size` parameter in the configs refers to the per-GPU batch size.





## Citation

If you find our code or paper useful, please consider citing
```bibtex
@inproceedings{giebenhain2024mononphm,
 author={Simon Giebenhain and Tobias Kirschstein and Markos Georgopoulos and  Martin R{\"{u}}nz and Lourdes Agapito and Matthias Nie{\ss}ner},
 title={MonoNPHM: Dynamic Head Reconstruction from Monocular Videos},
 booktitle = {Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
 year = {2024}}
```

If you find the NPHM dataset helpful, consider citing
```bibtex
@inproceedings{giebenhain2023nphm,
 author={Simon Giebenhain and Tobias Kirschstein and Markos Georgopoulos and  Martin R{\"{u}}nz and Lourdes Agapito and Matthias Nie{\ss}ner},
 title={Learning Neural Parametric Head Models},
 booktitle = {Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
 year = {2023}}
```


Contact [Simon Giebenhain](mailto:simon.giebenhain@tum.de) for questions, comments and reporting bugs, or open a GitHub Issue.

