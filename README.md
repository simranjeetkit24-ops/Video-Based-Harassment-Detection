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
| `explainability.py` | Generate final-layer attention maps and overlays for a video or the test set |
| `monitor.py` | Monitor a video with overlapping windows and harassment alerts |
| `monitor_live.py` | Monitor a webcam or video source in real time; facial emotion is context only |
| `monitor_multimodal.py` | Analyse a video with harassment scores and facial-emotion context |
| `plot_monitor_log.py` | Plot a saved monitoring log |
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
python explainability.py --video clip.mp4 --model both
```
Use a GPU (add `--fp16`) for the full 20-epoch setting. Trained weights are not
stored in the repo because of their size.

### Live camera monitoring

Run the live monitor from this folder to use the default camera:

```powershell
python monitor_live.py --source 0
```

Press `q` to stop. The monitor reports the video model's harassment probability
and may raise a threshold-based alert; it is a prototype for human review, not
a validated real-world safety decision system. Facial-emotion estimates are
shown only as context and do not determine alerts. On a CPU, analysis may lag
behind the live video.

### Attention visualizations

`explainability.py` loads the existing checkpoints from
`outputs/<model>/best_model`; it does not retrain or modify them. It uses the
same deterministic frame sampling, 224x224 resize, and processor
normalization as `predict.py`. The script saves original sampled frames,
attention heatmaps, overlays, a per-model montage, and a JSON summary under
`outputs/explainability/<model>/<video-name>/`. When both models are run, a
top-frame comparison is also saved under `outputs/explainability/comparisons/`.

```powershell
python explainability.py --video "path\to\clip.mp4" --model timesformer
python explainability.py --video "path\to\clip.mp4" --model videomae --no-display
python explainability.py --all-test --model both --no-display
```

TimeSformer visualizations use final-layer spatial CLS-to-patch attention and
temporal attention for frame scores. VideoMAE visualizations use final-layer
eager attention, grouped by its two-frame tubelets (the same map is shown for
both frames in each tubelet). These attention maps show model token
interactions; they are not causal explanations and do not establish why a
prediction was made. Running `--all-test` performs inference on every test
clip and may take a long time, especially on CPU.