cd ~/GPT-home-private/scheduler/
source activate pytorch_p38
case=$1
rank=$2
world_size=$3


python generate_heterogeneous_tc.py --case $case --rank $rank --nodes $world_size