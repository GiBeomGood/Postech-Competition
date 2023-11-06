#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

conda activate torch
python ./errors_prediction.py

conda activate kwon
python ./output_prediction.py