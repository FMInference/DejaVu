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
sudo apt-get -y install cuda-11-1

# Install Python package that is needed.
sudo apt install python3-pip -y
pip3 install cupy-cuda111==8.6.0
# pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install "pybind11[global]"
pip3 install transformers
pip install deepspeed==0.6.7

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

# setup ssh free:
ssh-copy-id -i ~/.ssh/id_rsa.pub fsuser@216.153.60.96

# It is necessary to reboot and make the installation effective.
sudo reboot