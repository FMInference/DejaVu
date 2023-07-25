for l in $(seq 0 8 64)
do  
    (trap 'kill 0' SIGINT; \
    CUDA_VISIBLE_DEVICES=0 python main_mlp.py --dataset c4 --lr 0.001 --L ${l} > logs/c4_mlp_out_${l}.txt & \
    CUDA_VISIBLE_DEVICES=1 python main_mlp.py --dataset c4 --lr 0.001 --L $((l+1)) > logs/c4_mlp_out_$((l+1)).txt & \
    CUDA_VISIBLE_DEVICES=2 python main_mlp.py --dataset c4 --lr 0.001 --L $((l+2)) > logs/c4_mlp_out_$((l+2)).txt & \
    CUDA_VISIBLE_DEVICES=3 python main_mlp.py --dataset c4 --lr 0.001 --L $((l+3)) > logs/c4_mlp_out_$((l+3)).txt & \
    CUDA_VISIBLE_DEVICES=4 python main_mlp.py --dataset c4 --lr 0.001 --L $((l+4)) > logs/c4_mlp_out_$((l+4)).txt & \
    CUDA_VISIBLE_DEVICES=5 python main_mlp.py --dataset c4 --lr 0.001 --L $((l+5)) > logs/c4_mlp_out_$((l+5)).txt & \
    CUDA_VISIBLE_DEVICES=6 python main_mlp.py --dataset c4 --lr 0.001 --L $((l+6)) > logs/c4_mlp_out_$((l+6)).txt & \
    CUDA_VISIBLE_DEVICES=7 python main_mlp.py --dataset c4 --lr 0.001 --L $((l+7)) > logs/c4_mlp_out_$((l+7)).txt & \
    wait)
done
