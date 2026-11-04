# Trains ONLY the two best models from the paper (TimeSformer, VideoMAE), then builds results.
# Usage (PowerShell, from the SHV_Implementation folder):   .\run_all.ps1
$ErrorActionPreference = "Stop"

# 1) TimeSformer: continue from your saved epoch-2 model, 3 more epochs, first 6 blocks frozen
python train_transformers.py --model timesformer --model_path outputs\timesformer\best_model --epochs 3 --freeze_blocks 6

# 2) VideoMAE: 4 epochs, first 8 blocks frozen
python train_transformers.py --model videomae --epochs 4 --freeze_blocks 8

python plot_results.py