# Install Cuda
sudo apt clean
sudo apt update
sudo apt purge nvidia-*  -y
sudo apt autoremove -y
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt clean
sudo apt update
sudo apt autoremove -y
sudo apt-get -y install cuda-11-3

sudo reboot

export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

sudo apt install python3-pip -y
# Install Cupy
pip3 install cupy-cuda11x==11.0.0
python3 -m cupyx.tools.install_library --cuda 11.x --library nccl
#optional for cupy
python3 -m cupyx.tools.install_library --cuda 11.x --library cutensor
python3 -m cupyx.tools.install_library --cuda 11.x --library cudnn

# Install PyTorch
pip3 install --pre torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

pip3 install "pybind11[global]"
pip3 install transformers
pip3 install deepspeed==0.6.7
pip3 install sentencepiece

# Install Ninja.
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force

# Install apex, from source. Do not use pip install for apex, it does not include amp_C
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ~

# Clone my repo:
git clone https://github.com/BinhangYuan/GPT-home-private.git
git config credential.helper 'cache --timeout=30000'
cd ~
git clone https://github.com/BinhangYuan/tc_cluster_setting.git
git config credential.helper 'cache --timeout=30000'
cd ~

# Follow this link to fix the installation on Datacrunch (Remember to use the instance without any cuda+docker)
# https://github.com/pytorch/pytorch/issues/35710#