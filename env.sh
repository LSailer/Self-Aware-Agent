
conda create -n self_aware_agent python=3.10

conda activate self_aware_agent


conda install conda-forge::pybullet -y
conda install conda-forge::matplotlib -y
#https://pytorch.org/get-started/locally/ 
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y
conda install conda-forge::opencv -y
conda install pytest -y
conda install -c conda-forge --name self_aware_agent tensorboard -y