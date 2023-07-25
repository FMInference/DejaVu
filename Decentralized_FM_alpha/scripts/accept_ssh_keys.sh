source ./ip_list.sh

for ip in "${ips[@]}"
do
  echo "Add "$ip" in ssh known hosts"
  ssh-keyscan -H "$ip" >> ~/.ssh/known_hosts
done