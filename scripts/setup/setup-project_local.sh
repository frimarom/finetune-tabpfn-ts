#!/usr/bin/env bash
/bin/bash ./clone_finetuning-scripts.sh

pip3 install --upgrade pip setuptools wheel

if [ ! -d "..external/finetune_tabpfn_v2" ]; then
  cd ../external/finetune_tabpfn_v2 || exit
  pip3 install -e .
  cd ../../ || exit
fi

pip3 install tabpfn-time-series==1.0.7 # install tabpfn-time-series since it cant be installed via requirements.txt
pip3 install -r requirements.txt # install other dependencies