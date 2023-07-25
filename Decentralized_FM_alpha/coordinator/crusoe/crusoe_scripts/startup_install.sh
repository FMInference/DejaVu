token=$1
coordinator_server_ip=$2

git clone https://"${token}"@github.com/BinhangYuan/GPT-home-private.git
cd ~/GPT-home-private/coordinator/crusoe
python3 crusoe_coordinator_vm_client.py --message "Checkout repo: done." --coordinator-server-ip $coordinator_server_ip

sudo apt-get update
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
sudo apt-get -y install python3-pip


cd ~/GPT-home-private/coordinator/crusoe
python3 crusoe_coordinator_vm_client.py --message "Install CUDA: done." --coordinator-server-ip $coordinator_server_ip

pip3 install cupy-cuda11x==11.0.0
python3 -m cupyx.tools.install_library --cuda 11.x --library nccl
python3 -m cupyx.tools.install_library --cuda 11.x --library cutensor
python3 -m cupyx.tools.install_library --cuda 11.x --library cudnn
pip3 install --pre torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install "pybind11[global]"
pip3 install transformers
pip3 install deepspeed==0.6.7
pip3 install sentencepiece
pip3 install flask

cd ~/GPT-home-private/coordinator/crusoe
python3 crusoe_coordinator_vm_client.py --message "Install Python Libs: done." --coordinator-server-ip $coordinator_server_ip