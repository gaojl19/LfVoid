#!/bin/bash

ROOT_DIR=$(pwd)

pip install -r requirements.txt

if [ ! -d "./outputs" ];then
mkdir outputs
fi

# install dreambooth
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .

cd examples/dreambooth
pip install -r requirements.txt

# copy the dreambooth code
cd $ROOT_DIR
cp diffusers/examples/dreambooth/train_dreambooth.py train_dreambooth.py