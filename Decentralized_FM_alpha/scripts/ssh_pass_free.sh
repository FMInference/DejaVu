source ./ip_list.sh

rm ./ds_pwfree_ssh/id_rsa.pub

echo "Issue command in "${ips[0]}" to generate rsa_key"
ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[0]}" "bash -s" < ./local_scripts/local_ssh_key_gen.sh

echo "Download key file from Rank-0 node."
scp -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[0]}":~/.ssh/id_rsa.pub ./ds_pwfree_ssh/

for rank in "${!ips[@]}"
do
  echo "Upload key file"
  scp -i ../binhang_ds3_aws_oregon.pem ./ds_pwfree_ssh/id_rsa.pub ubuntu@"${ips[rank]}":~/.ssh/
done


for rank in "${!ips[@]}"
do
  echo "Issue command in "${ips[rank]}" to generate setup ssh"
  ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/local_ssh_key_setup.sh
done


echo "Issue command in "${ips[0]}" to accept hostnames"
for hostname in "${hostnames[@]}"
do
  echo "Add "$hostname" in ssh known hosts"
  ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[0]}" "ssh-keyscan -H "$hostname" >> ~/.ssh/known_hosts"
done
