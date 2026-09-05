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

Build in progress — 05 Sep 2026.
