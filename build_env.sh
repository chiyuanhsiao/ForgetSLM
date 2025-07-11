#!/bin/bash

#   Copyright 2025 Chi-Yuan Hsiao (蕭淇元)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

source ~/.bashrc
conda create -p ./slm_env python=3.10
conda activate ./slm_env

git clone https://github.com/facebookresearch/seamless_communication
cd seamless_communication

pip install .
conda install -c conda-forge libsndfile==1.0.31

cd ..
pip install -r requirements.txt


