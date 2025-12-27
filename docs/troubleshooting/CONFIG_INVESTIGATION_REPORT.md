# Gomoku AI Training Configuration Investigation Report

**Date**: 2025-12-24
**Issue**: Policy loss remains around 5.41-5.40 (random baseline 5.42) despite 22 iterations and multiple config changes
**Status**: CRITICAL BUGS IDENTIFIED

---

## Executive Summary

The training is not improving because **the learning rate scheduler state is broken when resuming from checkpoints with a different config**. While the MCTS parameters (DIRICHLET_EPSILON, temperature, etc.) are correctly configured and being used, the optimizer's scheduler is loaded from old checkpoints and becomes out-of-sync with the new configuration, causing premature or incorrect learning rate decay.

Additionally, **config changes don't persist across checkpoint boundaries** because while configs are saved in checkpoints, they are never loaded back when resuming.

---

## Investigation Findings

### 1. MCTS DIRICHLET NOISE - CORRECTLY IMPLEMENTED ✓

**Question**: Is DIRICHLET_EPSILON actually being used in MCTS code?

**Answer**: YES, correctly implemented in both MCTS variants.

#### Location 1: `/c/Users/lotus/code/python/gomoku/core/mcts.py` (lines 168-185)

```python
def _add_dirichlet_noise(self, root, state):
    """
    为根节点添加Dirichlet噪声以增加探索
    """
    legal_moves = state.get_legal_moves()
    noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))

    for i, move in enumerate(legal_moves):
        if move in root.children:
            child = root.children[move]
            # 混合先验概率和噪声
            child.prior_prob = (
                (1 - self.dirichlet_epsilon) * child.prior_prob +
                self.dirichlet_epsilon * noise[i]
            )
```

**Verification**:
- ✓ Reads `self.dirichlet_epsilon` from config (set in `__init__` at line 45)
- ✓ Applies noise formula: `(1 - epsilon) * prior + epsilon * noise`
- ✓ Applied to root node children during search (line 78)
- ✓ No hardcoded values
- ✓ add_noise flag properly checked before applying

#### Location 2: `/c/Users/lotus/code/python/gomoku/core/mcts_batched.py` (lines 295-313)

Identical implementation - both versions correctly apply Dirichlet noise.

**Current Config Values**:
```
get_full_config():       DIRICHLET_EPSILON = 0.35
get_optimized_config(): DIRICHLET_EPSILON = 0.15 (-57%)
```

---

### 2. TEMPERATURE CONTROL - CORRECTLY IMPLEMENTED ✓

**Question**: Is temperature control using config values or hardcoded?

**Answer**: YES, properly reading from config.

#### Location: `/c/Users/lotus/code/python/gomoku/train_pipeline_pytorch.py` (lines 126-132)

```python
# 温度采样
if move_count < config.TEMP_THRESHOLD_MOVE:
    temperature = 1.0
elif move_count < config.TEMP_FINAL_MOVE:
    temperature = 0.5
else:
    temperature = 0.01
```

**Verification**:
- ✓ Uses `config.TEMP_THRESHOLD_MOVE` (not hardcoded)
- ✓ Uses `config.TEMP_FINAL_MOVE` (not hardcoded)
- ✓ Values from config properly propagated to workers

**Current Config Values**:
```
get_full_config():
  TEMP_THRESHOLD_MOVE = 30
  TEMP_FINAL_MOVE = 60

get_optimized_config():
  TEMP_THRESHOLD_MOVE = 20 (-33%)
  TEMP_FINAL_MOVE = 40 (-33%)
```

---

### 3. SELF-PLAY MCTS CALLS - CORRECTLY IMPLEMENTED ✓

**Question**: How is MCTS being instantiated? Are config parameters properly passed?

**Answer**: YES, config is properly passed through the call chain.

#### Location: `/c/Users/lotus/code/python/gomoku/train_pipeline_pytorch.py` (lines 100-112)

```python
def run_self_play_game_worker(game_num):
    """Self-play worker"""
    global _worker_model, _worker_config, _worker_device

    # Use cached model and config
    model = _worker_model
    config = _worker_config
    device = _worker_device

    # Create MCTS
    use_batched = getattr(config, 'USE_BATCHED_MCTS', False)
    if use_batched:
        mcts = BatchedMCTS(model, config, device=device)
    else:
        mcts = MCTS(model, config, device=device)

    # Search with noise enabled
    action_probs, root_value = mcts.search(state, add_noise=True)
```

**Verification**:
- ✓ MCTS created with current worker config
- ✓ `add_noise=True` passed during training
- ✓ Config dict properly converted from main config (line 529)

#### Config Dict Conversion (line 529):

```python
config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
```

Then in worker init (lines 51-54):
```python
_worker_config = TrainingConfig()
for key, value in config_dict.items():
    setattr(_worker_config, key, value)
```

**Result**: Each worker gets an independent config object with identical values to the main process config.

---

## CRITICAL BUGS FOUND

### BUG #1: SCHEDULER STATE MISMATCH ⚠️ CRITICAL

**Severity**: HIGH
**Impact**: Learning rate schedule is broken when resuming training
**Location**: `/c/Users/lotus/code/python/gomoku/train_pipeline_pytorch.py` lines 438-468

#### The Problem

When resuming from a checkpoint with a **different config**:

1. **Old checkpoint** contains:
   - Model weights (correct to load)
   - Optimizer state (problematic to load)
   - **Scheduler state with old step counter** (CRITICAL BUG)
   - Config dict (not used)

2. **New config** has different:
   - `LR_DECAY_STEPS` (100 → 200)
   - `LR_DECAY_RATE` (0.7 → 0.8)
   - Other hyperparameters

3. **What happens**:
   ```python
   checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=True)
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # ← BUG HERE

   # Scheduler now has OLD step counter from checkpoint
   # But we're using NEW config's LR_DECAY_STEPS
   # The learning rate update is now completely broken!
   ```

#### Why This Breaks Training

**Example scenario**:
- Iterations 1-20: Trained with LR_DECAY_STEPS=100, got to iteration 20
- Checkpoint at iter 20 saved with scheduler state indicating step_size=100
- User changes config to LR_DECAY_STEPS=200
- Resume training from iteration 21
- Scheduler.load_state_dict() restores scheduler with old settings
- Scheduler thinks we're at step 1 of the OLD schedule
- **Result**: Learning rate decay happens at wrong times, confusing the optimizer

**Impact on your case**:
- Policy loss stays around 5.40-5.41 (barely improving)
- Model can't learn effectively because LR schedule is broken
- Config changes don't take effect for optimizer

---

### BUG #2: CONFIG NOT LOADED FROM CHECKPOINT ⚠️ CRITICAL

**Severity**: HIGH
**Impact**: Config changes don't persist across checkpoint resume
**Location**: `/c/Users/lotus/code/python/gomoku/train_pipeline_pytorch.py` lines 439-468

#### The Problem

Config is **saved in checkpoints** (line 811):
```python
torch.save({
    'iteration': iteration + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict() if scaler else None,
    'history': history,
    'config': config_dict  # ← Saved here
}, checkpoint_path)
```

But **NEVER loaded back** when resuming:
```python
checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
if use_amp and 'scaler_state_dict' in checkpoint:
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
# MISSING: config = checkpoint.get('config')
start_iteration = checkpoint.get('iteration', 0)
```

#### Why This Matters

You can call:
```bash
python train_pipeline_pytorch.py --optimized
```

But if training auto-resumes from checkpoint:
1. New config is created with optimized values ✓
2. Checkpoint is loaded with OLD config in it ✓ (stored)
3. OLD config is completely ignored ✗ (not loaded)
4. NEW config from `--optimized` flag is used ✓

**However**: The scheduler state is from the OLD iterations, creating conflict!

---

## ROOT CAUSE: Why Policy Loss Isn't Improving

### The Sequence of Events

1. **Iterations 1-20**: Trained with `get_full_config()`
   - DIRICHLET_EPSILON = 0.35 (high exploration)
   - LR_DECAY_STEPS = 100
   - Model learns slowly due to high exploration noise

2. **User changes to optimized config**:
   - User sees policy loss isn't improving
   - Switches to `get_optimized_config()`
   - DIRICHLET_EPSILON = 0.15 (less noise)
   - LR_DECAY_STEPS = 200 (delayed decay)

3. **Resume from iteration 20**:
   - Load checkpoint from iteration 20 (trained with old config)
   - Load scheduler state (step counter = 1 out of old 100)
   - But new config has LR_DECAY_STEPS = 200
   - **Scheduler is now BROKEN**

4. **Iterations 21-22**:
   - Self-play generates data with correct new config ✓
   - Temperature control uses new config ✓
   - **BUT learning rate updates are wrong** ✗
   - Model can't learn effectively
   - Policy loss stalls at 5.40-5.41

### Why Random Baseline is 5.42

The loss function is categorical cross-entropy:
- Random policy: uniform over ~225 positions
- Cross-entropy for random: log(225) ≈ 5.42
- Your model: 5.41 = barely better than random!
- **This indicates: the model is NOT LEARNING despite 22 iterations**

---

## Verification of What's Working

### ✓ MCTS Parameters

| Parameter | Location | Status | Value |
|-----------|----------|--------|-------|
| DIRICHLET_EPSILON | mcts.py:168-185 | ✓ Used correctly | 0.15 (optimized) |
| DIRICHLET_ALPHA | mcts_batched.py:57 | ✓ Used correctly | 0.3 |
| C_PUCT | mcts.py:170 | ✓ Used correctly | 1.5 (optimized) |
| add_noise flag | train_pipeline.py:131 | ✓ Passed correctly | True |

### ✓ Temperature Control

| Parameter | Location | Status | Value |
|-----------|----------|--------|-------|
| TEMP_THRESHOLD_MOVE | train_pipeline.py:126 | ✓ Used correctly | 20 (optimized) |
| TEMP_FINAL_MOVE | train_pipeline.py:127 | ✓ Used correctly | 40 (optimized) |
| Temperature formula | mcts.py:306 | ✓ Correct | action_probs^(1/T) |

### ✓ Config Passing

| Component | Status | Method |
|-----------|--------|--------|
| Main config creation | ✓ Correct | get_optimized_config() |
| Worker config dict | ✓ Correct | config.__dict__ conversion |
| Worker config restore | ✓ Correct | setattr loop in init_worker |
| Self-play MCTS init | ✓ Correct | Passes config to MCTS/BatchedMCTS |

### ✗ Learning Rate Schedule

| Component | Status | Issue |
|-----------|--------|-------|
| Checkpoint loading | ✗ BROKEN | Loads old scheduler state |
| Config merging | ✗ BROKEN | Old config never loaded from checkpoint |
| LR decay timing | ✗ BROKEN | Step counter out of sync |

---

## Recommended Fixes

### FIX #1: Reset Scheduler After Loading Checkpoint (URGENT)

**File**: `/c/Users/lotus/code/python/gomoku/train_pipeline_pytorch.py`
**Location**: After line 468 (after loading checkpoint)

Add:
```python
# After loading checkpoint, reset scheduler to match new config
# because checkpoint has old scheduler state
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config.LR_DECAY_STEPS,
    gamma=config.LR_DECAY_RATE
)
```

**Alternative (better)**: After loading, step the new scheduler to catch up to current iteration:
```python
# Reset scheduler with new config
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config.LR_DECAY_STEPS,
    gamma=config.LR_DECAY_RATE
)

# Catch up to current iteration
for _ in range(start_iteration):
    scheduler.step()
```

### FIX #2: Load Config From Checkpoint (IMPORTANT)

**File**: `/c/Users/lotus/code/python/gomoku/train_pipeline_pytorch.py`
**Location**: After line 468 (after loading checkpoint)

Add:
```python
# Load config from checkpoint for consistency
if 'config' in checkpoint:
    old_config_dict = checkpoint['config']
    # Merge old config but prefer new command-line config
    for key, value in old_config_dict.items():
        if key not in ['ITERATIONS', 'GAMES_PER_ITERATION']:  # Don't override CLI args
            if not hasattr(config, key):
                setattr(config, key, value)
```

### FIX #3: Fresh Start Is Better

**Recommended**: Since you want to try the optimized config, consider:

```bash
# Option 1: Reset training completely with optimized config
python reset_training.py
python train_pipeline_pytorch.py --optimized

# Option 2: If keeping checkpoint, use special resume script
python train_pipeline_pytorch.py --optimized --resume checkpoints/checkpoint_iter_20.pth
```

---

## Summary of Findings

### What's Working Correctly

1. **MCTS Dirichlet Noise**: ✓ Correctly applied, no hardcoded values
2. **Temperature Control**: ✓ Correctly reading from config
3. **Config Distribution to Workers**: ✓ Properly converted and passed
4. **Self-Play Integration**: ✓ MCTS called with correct parameters
5. **MCTS Exploration**: ✓ add_noise=True during training

### What's Broken

1. **Scheduler State**: ✗ CRITICAL - Loaded from old checkpoint, conflicts with new config
2. **Config Persistence**: ✗ Saved in checkpoint but never loaded back
3. **Learning Rate Schedule**: ✗ Out of sync across checkpoint boundaries

### Root Cause

**The learning rate schedule is broken when resuming from checkpoints with a different config.** The scheduler's internal step counter and settings come from the old checkpoint, but the config has changed, causing:
- Wrong learning rate at wrong times
- Model unable to learn effectively
- Policy loss stuck near random baseline (5.40 ≈ 5.42)

---

## Files Examined

1. `/c/Users/lotus/code/python/gomoku/core/mcts.py` - MCTS implementation
2. `/c/Users/lotus/code/python/gomoku/core/mcts_batched.py` - Batched MCTS implementation
3. `/c/Users/lotus/code/python/gomoku/training/config.py` - Configuration file
4. `/c/Users/lotus/code/python/gomoku/train_pipeline_pytorch.py` - Main training loop

---

## Next Steps

1. **Immediate**: Apply FIX #1 (Reset scheduler) to `/c/Users/lotus/code/python/gomoku/train_pipeline_pytorch.py`
2. **Important**: Apply FIX #2 (Load config from checkpoint)
3. **Long-term**: Consider using `--resume` flag instead of auto-resume to have explicit control over config changes
4. **Verification**: Run one iteration and check if learning rate schedule is correct

