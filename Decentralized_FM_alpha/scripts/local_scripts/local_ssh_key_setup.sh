cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
sudo chmod -R 700 .ssh
sudo chmod -R 640 .ssh/authorized_keys
