DELAY_MS=$1
if [ $# -eq 2 ]
then
  NET_INF=$2
  echo "TC set up delay to $1 ms."
  sudo tc qdisc add dev $NET_INF root netem delay ${DELAY_MS}ms
else
  echo "TC set up delay to $1 ms."
  sudo tc qdisc add dev ens3 root netem delay ${DELAY_MS}ms
fi