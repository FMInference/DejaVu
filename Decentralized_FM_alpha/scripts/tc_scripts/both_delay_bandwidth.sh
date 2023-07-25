DELAY_MS=$1
RATE_GBIT=$2

if [ $# -eq 3 ]
then
  NET_INF=$3
  echo "TC set up delay to $1 ms; bandwidth to $2 Gbit."
  LIMIT_PKTS=$(echo "$RATE_GBIT * 15000 * $DELAY_MS * 1.5" | bc -q)
  sudo tc qdisc add dev $NET_INF root netem delay ${DELAY_MS}ms rate ${RATE_GBIT}Gbit limit ${LIMIT_PKTS}
else
  echo "TC set up delay to $1 ms; bandwidth to $2 Gbit."
  LIMIT_PKTS=$(echo "$RATE_GBIT * 15000 * $DELAY_MS * 1.5" | bc -q)
  sudo tc qdisc add dev ens3 root netem delay ${DELAY_MS}ms rate ${RATE_GBIT}Gbit limit ${LIMIT_PKTS}
fi