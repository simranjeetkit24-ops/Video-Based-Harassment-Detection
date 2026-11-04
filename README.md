# Video-Based Harassment Detection using Fine-Tuned Video Transformers

Minor project by **Simranjeet Kaur** (Roll No. 24124108), under the guidance of
**Dr. Nitika Nigam**, NIT Jalandhar.

Binary classification of short video clips as **Harassment / Non-Harassment** by
fine-tuning Kinetics-400 pretrained **TimeSformer** and **VideoMAE** on the SHV dataset.
Based on:
- Mahmud, Islam, Hossain, *Video-Based Sexual Harassment Detection Using Fine-Tuned Video Transformers*, arXiv:2610.15354, 2026.
- Dhuri and Patil, *Crime prediction against women in surveillance videos using deep learning models: A review*, Computers and Electrical Engineering 128, 2025.

## Results (181 test videos)

| Model | Accuracy | Macro F1 | Harassment precision | Harassment recall |
|---|---|---|---|---|
| TimeSformer | 91.71% | 0.916 | 0.961 | 0.859 |
| VideoMAE | 91.71% | 0.916 | 0.949 | 0.871 |
| Simple CNN from scratch (`train.py`) | about 51% (validation) | n/a | n/a | n/a |

Figures: `outputs/confusion_matrices.png`, `outputs/training_curves.png`, `outputs/comparison_table.md`.

The two transformers tie (difference of one video), so neither is claimed better.
The scratch CNN stays near chance, showing the value of pretraining on a small dataset.

## Training setup and honest notes

- Both models start from Kinetics-400 weights; the last layer is replaced by a 2-class head.
- Because training was done on a laptop CPU, the early blocks were frozen
  (TimeSformer: first 6 of 12, VideoMAE: first 8 of 12) and trained for 3 to 4 epochs
  (the paper uses 20).
- 10% of the training folder is held out as validation to choose the best epoch;
  the test folder is used once at the end (the paper monitored the test set).
- Dataset is small and YouTube-sourced; results may not transfer to real CCTV footage.

## Files

| File | Purpose |
|---|---|
| `check_dataset.py` | Counts videos per class and split |
| `dataset.py`, `train.py` | Simple CNN baseline trained from scratch |
| `train_transformers.py` | Fine-tune TimeSformer / VideoMAE (main script) |
| `evaluate_saved.py` | Evaluate a saved model on the test folder |
| `predict.py` | Classify a single video |
| `plot_results.py` | Comparison table, curves, confusion matrices |
| `baselines.py` | CNN-LSTM / CNN-BiGRU / 3D-ResNet code (written but **not run or evaluated**) |
| `run_all.ps1` | Runs the two-model training |

## Setup

```
pip install -r requirements.txt
```

## Dataset

Not included in this repo. Download the SHV dataset from Zenodo
(https://doi.org/10.5281/zenodo.22653003) and arrange it as:

```
Sexual_Harassment_Dataset/
  train/Harassment  train/Non_Harassment
  test/Harassment   test/Non_Harassment
```
Set the path with `--root` or edit `DEFAULT_ROOT` in `train_transformers.py`.

## Run

```
python train_transformers.py --model timesformer --epochs 3 --freeze_blocks 6
python train_transformers.py --model videomae --epochs 4 --freeze_blocks 8
python plot_results.py
python evaluate_saved.py --model timesformer
python predict.py --video clip.mp4 --model timesformer
```
Use a GPU (add `--fp16`) for the full 20-epoch setting. Trained weights are not
stored in the repo because of their size.