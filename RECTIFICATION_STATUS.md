# Rectification Status (2026-05-17 UTC)

This file records the concrete rectifications requested in conversation.

## 1) Naming status
- Repository top-level name in README is **Visual Execution Model (VEM)**.

## 2) Restoration status (important files)
The following previously removed files are restored and present:
- `puzzle_dataset.py`
- `pretrain.py`
- `evaluate.py`
- `dataset/build_arc_dataset.py`
- `dataset/build_maze_dataset.py`
- `models/hrm/hrm_act_v1.py`
- `models/sparse_embedding.py`
- `config/cfg_pretrain.yaml`
- `config/arch/hrm_v1.yaml`
- `puzzle_visualizer.html`
- `arc_eval.ipynb`

## 3) Integration and conflict checks rerun
- Merge conflict marker scan
- Python compile sanity
- Integration smoke (`check_integrations.py`)
- World-model eval smoke
- Perception eval smoke

## 4) Address map status
- `CODE_ADDRESS_INDEX.md` is synchronized to current repository layout and line-level anchors.

## 5) Mistake rectification summary
- Reversal of accidental over-deletion has been completed.
- Current branch preserves both VEM integration work and restored legacy components.
