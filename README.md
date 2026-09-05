# Drone Anomaly Cascade

Real-time video anomaly detection for drone footage, built for AHC's Visual Intelligence Hackathon (05 Sep 2026).

## Problem

Drones flying over cities capture mostly routine footage, but a small fraction contains events that need a response while the drone is still overhead — accidents, fires, congestion, stalled vehicles, and similar. Standard object detectors can't tell whether something counts as an anomaly, since that depends on context rather than the object class alone. Vision-language models can reason about context in language, but large VLMs are too slow and expensive to run continuously across many live drone feeds.

## Approach: cascade

- **Stage 1 (always-on, cheap):** a lightweight motion/heuristic filter runs on every frame and flags candidate segments (e.g. a vehicle stationary too long, a motion burst, an unusual color signature).
- **Stage 2 (verifier, small VLM):** a fine-tuned small vision-language model runs only on flagged segments and decides the anomaly class and confidence, using context rather than a fixed object list.
- Only the small VLM runs at inference time. Larger hosted models are used solely for prototyping and generating training/distillation data during development, per the hackathon's runtime constraint.

## Label set

`normal`, `traffic_accident`, `traffic_congestion`, `stalled_or_broken_down_vehicle`, `vehicle_blocking_traffic`, `wrong_way_driving`, `road_spill_or_debris`, `waterlogging_or_flood`, `fire`, `smoke`, `fighting_or_violence`, `loitering_or_suspicious_presence`

## Status

Paused 05 Sep 2026 (Colab free-tier GPU quota exhausted after an OOM crash mid-fine-tune). Treating this as an ongoing project rather than racing the hackathon clock — resume whenever GPU access is back (retry Colab later, or fix Kaggle's GPU lock by verifying phone number under account Settings).

**Done so far:**
- Dataset (train + public test, 12 classes) pulled into a Colab environment via a Drive-shortcut copy (public gdown links hit Google's per-file quota wall under hackathon-wide load — the shortcut route sidesteps that).
- Confirmed real CSV schema on this data cut: `ground_truth.csv` has no `level` column (contrary to the dataset doc) — tier is inferred instead from whether `start_time_sec`/`end_time_sec` are populated. `video_id` maps to a real file via `videos.csv`.
- Class counts (train): normal 973, traffic_accident 565, traffic_congestion 268, loitering_or_suspicious_presence 300, stalled_or_broken_down_vehicle 223, wrong_way_driving 164, road_spill_or_debris 151, vehicle_blocking_traffic 148, waterlogging_or_flood 95, smoke 85, fighting_or_violence 124, fire 77. Every anomaly class has 100% temporal (event-window) coverage; `normal` has none (expected).
- Working frame-extraction pipeline: per class, cap at N videos, pull frames from the padded event window (whole clip for `normal`), build chat-formatted (system/user-images/assistant-JSON) training examples for Unsloth vision SFT.
- Hit and fixed a real OOM: building the training set as a plain list with all frames pre-decoded as PIL images (~8,000 images) crashed the Colab runtime. Fixed with `LazyVisionSFTDataset`, a `torch.utils.data.Dataset` that decodes images per-item instead of all at once.
- [stage1/](stage1/) — a classical-CV always-on filter (motion bursts, stopped-vehicle timing, crowd/loitering) using pretrained YOLOv8n, no training required. Not yet tested against real footage, and doesn't cover fire/smoke/flood/spill/wrong-way (those rely on Stage 2 for now).

**Next steps, in order:**
1. Get GPU access back (Colab quota reset, or Kaggle after phone verification).
2. Re-run: mount Drive → copy dataset → `find_root()`/classes → extraction cell (`records = [...]`) → `LazyVisionSFTDataset` cell. All of this is proven to work; it just needs re-running against a fresh runtime.
3. Before committing to a long training run: descope for a first working pass — cap ~60-80 videos/class (~700-900 total) and 3 frames/clip instead of 6, 1 epoch. Watch time-per-step once training starts; swap to a smaller model (e.g. Qwen2.5-VL-3B) if a 7B LoRA fine-tune proves too slow for the remaining time budget.
4. Load Qwen2.5-VL-7B-Instruct via Unsloth (4-bit, LoRA, frozen vision encoder) and fine-tune on `train_dataset`/`val_dataset`.
5. Save the LoRA adapter, run inference on the public test set, score against `test/ground_truth.csv`.
6. Only once Stage 2 works end-to-end: test [stage1/filter.py](stage1/filter.py) against real footage and wire the two stages together.
