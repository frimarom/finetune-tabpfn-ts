#!/usr/bin/env bash
/bin/bash ./clone_finetuning-scripts.sh

if [ ! -d "..external/finetune_tabpfn_v2" ]; then
  cd ../external/finetune_tabpfn_v2 || exit
  pip install -e .
  cd ../../ || exit
fi


pip install --upgrade pip setuptools wheel
pip install tabpfn-time-series==1.0.7 # install tabpfn-time-series since it cant be installed via requirements.txt
pip install -r requirements.txt # install other dependencies