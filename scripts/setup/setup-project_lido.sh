#!/usr/bin/env bash
/bin/bash ./clone_finetuning-scripts.sh

pip3 install --upgrade pip setuptools wheel

if [ ! -d "..external/finetune_tabpfn_v2" ]; then
  cd ../external/finetune_tabpfn_v2 || exit
  pip3 install -e .
  cd ../../ || exit
fi

# wishi washi because of lido... grr
pip3 install pyarrow=20.0.0 --only-binary=:all: # on a cluster pyarrow has to be installed via conda
pip3 install "datasets<4.0"
pip3 install tabpfn-time-series==1.0.7 --no-build-isolation --no-deps
pip3 install gluonts backoff tabpfn tabpfn-client tabpfn-common-utils python-dotenv

pip3 install -r requirements.txt # install other dependencies