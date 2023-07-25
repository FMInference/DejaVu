rank=$1
rm ~/.ssh/id_rsa_r*
ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
