# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14 --data_path /path/IDRiD --type eyes --k 1
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14 --data_path /path/IDRiD --type eyes --k 2
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14 --data_path /path/IDRiD --type eyes --k 3
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14 --data_path /path/IDRiD --type eyes --k 4
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14 --data_path /path/IDRiD --type eyes --k 5
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14 --data_path /path/IDRiD --type eyes --k 10

CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14@336px --data_path /path/IDRiD --type eyes --k 40

# grid search
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14@336px --data_path /path/IDRiD --type eyes --k 40  --tau_t 0.001
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14@336px --data_path /path/IDRiD --type eyes --k 40  --tau_t 0.005
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14@336px --data_path /path/IDRiD --type eyes --k 40  --tau_t 0.02
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14@336px --data_path /path/IDRiD --type eyes --k 40  --tau_t 0.03
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14@336px --data_path /path/IDRiD --type eyes --k 40  --tau_t 0.04
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14@336px --data_path /path/IDRiD --type eyes --k 40  --tau_t 0.05
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14@336px --data_path /path/IDRiD --type eyes --k 40  --tau_t 0.06
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14@336px --data_path /path/IDRiD --type eyes --k 40  --tau_t 0.07
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14@336px --data_path /path/IDRiD --type eyes --k 40  --tau_t 0.08
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14@336px --data_path /path/IDRiD --type eyes --k 40  --tau_t 0.09
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14@336px --data_path /path/IDRiD --type eyes --k 40  --tau_t 0.1
# CUDA_VISIBLE_DEVICES=1 python Main.py -a ViT-L/14@336px --data_path /path/IDRiD --type eyes --k 40  --tau_t 0.2