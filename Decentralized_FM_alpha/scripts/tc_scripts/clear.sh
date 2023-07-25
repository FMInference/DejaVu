if [ $# -eq 1 ]
then
  NET_INF=$1
  echo "TC clear all qdisc for ens3."
  sudo tc qdisc delete dev $NET_INF root
else
  echo "TC clear all qdisc for ens3."
  sudo tc qdisc delete dev ens3 root
fi