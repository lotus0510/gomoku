# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaZero-style Gomoku (Five-in-a-Row) AI training system using PyTorch for deep reinforcement learning. The system trains an AI from scratch through self-play without human game records.

**Language**: Traditional Chinese (繁體中文) for documentation and comments.

**Python Environment**: ⚠️ **CRITICAL** - This project uses `.venv` virtual environment (NOT `venv`). **ALWAYS activate before running**:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

**IMPORTANT**: The virtual environment directory is `.venv` (with dot prefix), not `venv`. Using the wrong path will cause import errors.

## ⚠️ CRITICAL WORKFLOW: Training Stop Logging

**When the user stops training and asks to record the stop reason:**

1. **Read current training state** from `checkpoints/training_history.json`
2. **Append a new record** to `docs/training_stop_log.json` (JSON format)
3. **Create new entry** in the `training_stops` array with auto-incremented ID
4. **Fill in all fields** (see structure below):
   - `timestamp` (ISO 8601 format with +08:00 timezone, e.g., "2025-12-26T16:30:00+08:00")
   - `stop_reason` (category, manual flag, description)
   - `training_progress` (current/total iterations, completion %, duration)
   - `hyperparameters` (all config values from `training/config.py`)
   - `metrics` (loss, value_network, policy_network, training)
   - `health_diagnosis` (healthy/warnings/critical arrays with status)
   - `problem_analysis` (main_problem, severity, timeline, root_causes)
   - `actions_taken` (action_type, adjustments)
   - `expected_results` (short/medium/long-term goals)
   - `related_files` (paths to checkpoints, configs, etc.)
   - `notes` (array of important observations)
   - `next_steps` (array of recommended actions)
   - `recorder` (always "Claude Code")
   - `reviewed` (boolean, default false)
5. **Update metadata**: Increment `total_stops` and update `last_updated` timestamp

**Format**: JSON with structured nested objects and arrays

**Template reference**: See existing examples in `docs/training_stop_log.json`

**Timezone**: Always use UTC+8 (Taiwan/Taipei time) in ISO 8601 format with +08:00 suffix

**DO NOT** create a new file - always **append** to the existing `training_stops` array in the JSON file.

## Core Commands

### Training

```bash
# Full training (1000 iterations, 100 games per iteration)
python train_pipeline_pytorch.py -i 1000 -g 100

# Quick test (5-10 minutes validation)
python train_pipeline_pytorch.py -f -i 20 -g 20

# Resume from checkpoint
python train_pipeline_pytorch.py --resume checkpoints/checkpoint_iter_50.pth

# Use optimized config (default, recommended)
python train_pipeline_pytorch.py  # Uses optimized config by default

# Use legacy config (for comparison only)
python train_pipeline_pytorch.py --legacy
```

### Analysis & Monitoring

```bash
# Generate comprehensive training charts (most commonly used)
python analysis/plot_comprehensive.py

# Real-time monitoring (run in separate terminal during training)
python analysis/monitor_training.py

# Training health diagnostics
python analysis/diagnose_training.py
```

### Playing & Evaluation

```bash
# Human vs AI
python play_vs_ai.py

# Watch AI self-play
python interactive/watch_ai.py

# Reset training environment (WARNING: deletes all checkpoints)
python reset_training.py
```

## Architecture Overview

### Training Pipeline Flow

```
1. Self-Play (Multi-process)
   - NUM_WORKERS processes generate games in parallel
   - Each worker has cached model for inference
   - Uses BatchedMCTS for 8x performance boost

2. Data Augmentation
   - 8-fold symmetry transformation (rotation + reflection)
   - Parallel processing with multiprocessing.Pool

3. Experience Replay
   - Prioritized replay buffer (400K-800K capacity)
   - TD-error based priority sampling
   - Beta annealing for importance sampling

4. Neural Network Training
   - Mixed precision (AMP) when GPU available
   - Dual-head output: Policy + Value
   - Gradient clipping + L2 regularization
   - StepLR scheduler with configurable decay

5. Checkpoint Management
   - Auto-save every iteration
   - Preserves all checkpoints by default
   - Custom reset strategy for value network collapse
```

### Key Components

#### 1. **Neural Network** (`core/neural_net.py`)
- **Architecture**: SE-ResNet (Squeeze-and-Excitation ResNet)
- **Depth**: 10 residual blocks (configurable)
- **Width**: 128 filters (configurable)
- **Input**: (batch, 3, 15, 15) - 3 channels: current player pieces, opponent pieces, current turn
- **Output**:
  - Policy head: (batch, 225) - action probabilities
  - Value head: (batch, 1) - position evaluation [-1, 1]

#### 2. **MCTS** (`core/mcts_batched.py`)
- **Batched MCTS**: Groups leaf node expansions for batch neural network inference
- **Virtual Loss**: Prevents thread collision in parallel search
- **Key Parameters**:
  - `C_PUCT`: Exploration constant (1.5-2.0)
  - `DIRICHLET_EPSILON`: Root noise for exploration (0.15-0.35)
  - `MCTS_BATCH_SIZE`: Batch size for inference (default 8)

#### 3. **Custom Reset Strategy** (`training/custom_reset_strategy.py`)
**Critical for preventing value network collapse** (MAE 0.77 → 1.04 U-curve issue).

Triggers when:
- Value Loss increases 5 consecutive times
- MAE worsening AND > 0.85 threshold
- After iteration 50 (safety check)

Actions:
- Rollback to iteration 40 checkpoint (healthy state)
- Clear latest 50% of replay buffer
- Adjust learning rate (1.5x boost)
- Max 3 resets per training run

#### 4. **Configuration System** (`training/config.py`)
Three configuration modes:
- **Optimized** (default): Recommended for production training
  - Lower exploration (DIRICHLET_EPSILON: 0.15)
  - Delayed LR decay (200 steps vs 100)
  - Larger buffer (800K vs 400K)
- **Full** (legacy): Original config with known issues
  - High exploration (0.35) causes training instability
- **Fast Test**: Quick validation (3 ResBlocks, 50 MCTS sims)

### Multiprocessing Architecture

**Worker Initialization** (`init_worker` in `train_pipeline_pytorch.py`):
- Each worker process loads model once at initialization
- Model cached in global `_worker_model` variable
- Avoids repeated model loading (major performance improvement)
- Critical for Windows: `multiprocessing.set_start_method('spawn', force=True)`

**Self-Play Workers**:
- Independent processes with own GameState
- Shared model weights (read-only)
- Return training data + game metadata to main process

### Critical Value Parameters

#### VALUE_GAMMA (Discount Factor)
Controls incentive for winning quickly:
- **0.995** (default): 18.2% reward difference between fast/slow wins
- **1.0**: No efficiency incentive (may cause prolonged games)
- **0.99**: Too aggressive (Edge-Rush risk)

Formula: `value = base_value × (VALUE_GAMMA ^ steps_to_end)`

#### Temperature Schedule
- **Steps 1-30**: temp=1.0 (exploration)
- **Steps 31-60**: linear decay (transition)
- **Steps 60+**: temp=0.01 (deterministic)

## Training Failure Patterns & Solutions

### 1. Value Network Collapse (U-shaped MAE)
**Symptoms**:
- MAE: 0.77 → 0.87 → 1.04
- Value std drops below 0.3
- Games become shorter (套路化/pattern collapse)

**Solution**: Custom reset strategy automatically triggers at `training/custom_reset_strategy.py`

### 2. Edge-Rush (边缘速胜策略)
**Symptoms**:
- Average game length < 20 steps
- Pieces concentrated on board edges (row 0 or 14)
- Top-1 policy probability > 0.5

**Solution**:
- Increase VALUE_GAMMA (0.99 → 0.995 or 0.998)
- Increase exploration (DIRICHLET_EPSILON)

### 3. Training Stagnation
**Symptoms**:
- Policy loss plateaus for 20+ iterations
- Gradient norm < 0.1 (gradient vanishing)
- Win rate stuck

**Solutions**:
- Check if LR decayed too early (use optimized config)
- Increase learning rate temporarily
- Clear old data from replay buffer

### 4. Gradient Issues
**Vanishing** (norm < 0.1):
- Increase learning rate
- Check for dead ReLUs
- Reduce network depth temporarily

**Exploding** (norm > 10):
- Reduce learning rate
- Check GRADIENT_CLIP_NORM setting
- Look for data anomalies

## Monitoring Health Metrics

**Six Core Indicators** (see `docs/troubleshooting/TRAINING_METRICS_GUIDE.md`):

1. **Policy Loss**: Should decrease from ~6.0 to ~4.5-5.0
2. **Value MAE**: Healthy range 0.70-0.85, critical if > 0.95
3. **Win Rate** (vs random): Should reach >85%
4. **Game Length**: Healthy 50-70 steps, warning if <40
5. **Gradient Norm**: Healthy 0.5-5.0
6. **Value Std**: Healthy >0.3, collapsed if <0.2

**Check after every 10 iterations**:
```bash
python analysis/diagnose_training.py
```

## File Organization

- **`core/`**: Game logic, MCTS, neural network
- **`training/`**: Configuration, replay buffer, custom strategies
- **`analysis/`**: Only 3 core tools (plot, monitor, diagnose)
- **`docs/`**: Comprehensive documentation (guides, troubleshooting, archived)
- **`checkpoints/`**: Models, training history, game logs
- **`interactive/`**: UI for playing and watching (includes arena.py, elo_rating.py for evaluation)
- **`logs/`**: Training stop logs, game records

**Important**:
- The project recently underwent major cleanup (2025-12-26). Root directory analysis scripts were moved to `analysis/` or deleted. Only essential tools remain.
- `evaluation/` directory was merged into `interactive/` - all evaluation tools (Arena, ELO) are now in `interactive/`
- Training pipeline imports: `from interactive.arena import Arena`

## Development Notes

### PyTorch vs TensorFlow
This is a **PyTorch implementation** (migrated from TensorFlow in v2.0.0). The `requirements.txt` still lists TensorFlow as legacy - ignore it. Install PyTorch instead:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Mixed Precision Training
Automatically enabled when CUDA available:
- Uses `torch.cuda.amp.GradScaler`
- ~40% memory reduction
- ~30% speed improvement

### Learning Rate Schedule
Critical understanding:
- Initial LR: 0.0001
- Decay every LR_DECAY_STEPS iterations
- Multiplier: LR_DECAY_RATE (0.8)
- Floor: MIN_LEARNING_RATE (1e-5)

**Optimized config delays decay** (100 → 200 steps) to allow better learning before convergence phase.

### Checkpoint Structure
```python
{
    'iteration': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'scaler_state_dict': dict,  # AMP scaler
    'history': dict,  # All training metrics
    'config': dict  # TrainingConfig snapshot
}
```

**Auto-resume**: Training automatically resumes from latest checkpoint if `training_history.json` exists.

### Config Change Detection
When resuming with modified config (e.g., LR_DECAY_STEPS changed), the pipeline:
1. Detects config mismatch
2. Reinitializes scheduler with new config
3. Syncs scheduler to current iteration
4. Prints warning about changed parameters

This prevents LR schedule corruption.

## Common Pitfalls

1. **Forgetting to use optimized config**: Default mode is now optimized, but be aware legacy mode exists
2. **Insufficient replay buffer**: Training fails if buffer size < batch size
3. **Too high exploration in late training**: DIRICHLET_EPSILON should be low (0.15) for stable learning
4. **Ignoring value network health**: MAE is the most critical metric - monitor closely
5. **Running out of GPU memory**: Reduce BATCH_SIZE or NUM_RES_BLOCKS/NUM_FILTERS
6. **Windows multiprocessing issues**: Must use `spawn` method (already handled in code)

## Documentation References

- **Quick Start**: `README.md`
- **Full Documentation Index**: `docs/INDEX.md`
- **Training Concepts**: `docs/guides/TRAINING_CONCEPTS.md`
- **Monitoring Guide**: `docs/guides/MONITORING_METRICS.md`
- **Developer Guide**: `docs/guides/DEVELOPER_GUIDE.md`
- **Custom Reset Strategy**: `docs/guides/CUSTOM_RESET_GUIDE.md` (highly recommended for value collapse issues)
- **Metrics Reference**: `docs/troubleshooting/TRAINING_METRICS_GUIDE.md`
- **Analysis Tools**: `analysis/README.md`

## Code Style Notes

- **Language**: All documentation, comments, and print statements are in Traditional Chinese
- **Naming**: Mix of English variable names with Chinese comments
- **File encoding**: UTF-8 with BOM for Chinese characters
- **Line length**: No strict limit, readability prioritized
- **Docstrings**: Mix of Chinese and parameter descriptions

When modifying code, maintain consistency with existing style - use Traditional Chinese for user-facing text and English for technical variable names.
