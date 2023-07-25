gpu_type=$1
source activate pytorch_p38


echo "Test dim 2048"
PYTHONPATH=../ python macro_benchmark_transformer_layer_run_time.py --embedding-dim 2048 --num-heads 16 >> "../logs/layer_time_${gpu_type}_d2048.log"

echo "Test dim 2560"
PYTHONPATH=../ python macro_benchmark_transformer_layer_run_time.py --embedding-dim 2560 --num-heads 32 >> "../logs/layer_time_${gpu_type}_d2560.log"

echo "Test dim 4096"
PYTHONPATH=../ python macro_benchmark_transformer_layer_run_time.py --embedding-dim 4096 --num-heads 32 >> "../logs/layer_time_${gpu_type}_d4096.log"

echo "Test dim 5160"
PYTHONPATH=../ python macro_benchmark_transformer_layer_run_time.py --embedding-dim 5160 --num-heads 40 >> "../logs/layer_time_${gpu_type}_d5160.log"

echo "Test dim 12288"
PYTHONPATH=../ python macro_benchmark_transformer_layer_run_time.py --embedding-dim 12288 --num-heads 96 >> "../logs/layer_time_${gpu_type}_d12288.log"