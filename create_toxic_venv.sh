#!/usr/bin/env bash

VENVNAME=toxic_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter
python -m ipykernel install --user --name=$VENVNAME
python -m nltk.downloader all

test -f requirements.txt && pip install -r requirements.txt
sudo apt install graphviz

deactivate
echo "build $VENVNAME"