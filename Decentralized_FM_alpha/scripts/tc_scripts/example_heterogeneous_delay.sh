sudo tc qdisc add dev ens3 root handle 1: prio
sudo tc qdisc add dev ens3 parent 1:1 handle 10: netem delay 10ms
sudo tc qdisc add dev ens3 parent 1:2 handle 20: netem delay 20ms
sudo tc filter add dev ens3 parent 1:0 protocol ip prio 1 u32 match ip dst 172.31.8.184/32 flowid 1:1
sudo tc filter add dev ens3 parent 1:0 protocol ip prio 1 u32 match ip src 172.31.8.184/32 flowid 1:1
sudo tc filter add dev ens3 parent 1:0 protocol ip prio 1 u32 match ip dst 172.31.10.3/32 flowid 1:2
sudo tc filter add dev ens3 parent 1:0 protocol ip prio 1 u32 match ip src 172.31.10.3/32 flowid 1:2