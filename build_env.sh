source ~/.bashrc
conda create -p ./slm_env python=3.10
conda activate ./slm_env

git clone https://github.com/facebookresearch/seamless_communication
cd seamless_communication

pip install .
conda install -c conda-forge libsndfile==1.0.31

cd ..
pip install -r requirements.txt


