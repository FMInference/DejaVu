'''
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A40          On   | 00000000:05:00.0 Off |                  Off |
|  0%   33C    P8    30W / 300W |      2MiB / 49140MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A40          On   | 00000000:06:00.0 Off |                  Off |
|  0%   34C    P8    31W / 300W |      2MiB / 49140MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A40          On   | 00000000:07:00.0 Off |                  Off |
|  0%   35C    P8    32W / 300W |      2MiB / 49140MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A40          On   | 00000000:08:00.0 Off |                  Off |
|  0%   35C    P8    31W / 300W |      2MiB / 49140MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA A40          On   | 00000000:09:00.0 Off |                  Off |
|  0%   33C    P8    31W / 300W |      2MiB / 49140MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA A40          On   | 00000000:0A:00.0 Off |                  Off |
|  0%   33C    P8    31W / 300W |      2MiB / 49140MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA A40          On   | 00000000:0B:00.0 Off |                  Off |
|  0%   34C    P8    32W / 300W |      2MiB / 49140MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA A40          On   | 00000000:0C:00.0 Off |                  Off |
|  0%   35C    P8    32W / 300W |      2MiB / 49140MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
'''

# pull code and download GLM-130B 
# https://github.com/THUDM/GLM-130B
# cat glm-130b-sat.tar.part_* > glm-130b-sat.tar
# tar xvf glm-130b-sat.tar

# install miniconda
# https://docs.conda.io/en/latest/miniconda.html

# install latest pytorch with cuda=11.6
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

# install GLM's dependency
pip install SwissArmyTransformer==0.2.12
pip install icetk
# pip install apex # this one is optional
pip install scipy
pip install dataclass_wizard
pip install cpm_kernels

cd GLM-130B
# edit the `CHECKPOINT_PATH'
vim configs/model_glm_130b.sh 
# edit hyperparameters
vim scripts/generate.sh

# run in interactive mode
bash scripts/generate.sh --input-source interactive
