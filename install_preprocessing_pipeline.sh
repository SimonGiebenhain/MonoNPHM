#!/bin/bash

cd src/mononphm/preprocessing/

# facer repository
pip install git+https://github.com/FacePerceiver/facer.git@main

# MICA
git clone git@github.com:Zielon/MICA.git
cd MICA
./install.sh
cd ..

# metrical tracker
git clone git@github.com:Zielon/metrical-tracker.git
cd metrical-tracker
./install.sh
# replace some file in the metrical tracker repository:
cp ../replacement_code/config.py configs/config.py
cp ../replacement_code/generate_dataset.py datasets/generate_dataset.py
cp ../replacement_code/tracker.py tracker.py
cd ..

# normal predictor
git clone git@github.com:boukhayma/face_normals.git
mkdir face_normals/pretrained_models/

# MODNet for image matting
git clone git@github.com:ZHKKKe/MODNet.git

# PIPnet
git clone https://github.com/jhb86253817/PIPNet.git
cd PIPNet/FaceBoxesV2/utils
sh make.sh
cd ../..
mkdir snapshots



#

