# SFT TQDM Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add visible tqdm progress bars to the long-running SFT stages `m1`, `m2`, `m3`, and `eval`.

**Architecture:** Introduce a small shared progress helper under `SFT/` so stage code can reuse consistent tqdm defaults without scattering imports. Use iterable progress bars for `m1`, `m2`, and `eval`, and a coarse stage-progress bar for `m3` to complement TRL's own training progress output.

**Tech Stack:** Python, tqdm, unittest, TRL

---

### Task 1: Shared progress wrapper

**Files:**
- Create: `SFT/progress.py`
- Test: `tests/test_progress_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_iter_progress_passes_desc_and_total():
    list(iter_progress([1, 2], desc="m1", total=2))
    progress_factory.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_progress_integration -v`
Expected: FAIL because `SFT.progress` does not exist

- [ ] **Step 3: Write minimal implementation**

```python
def iter_progress(iterable, *, desc, total=None):
    return tqdm(iterable, desc=desc, total=total)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_progress_integration -v`
Expected: PASS


### Task 2: M1/M2/eval integration

**Files:**
- Modify: `SFT/data_builder.py`
- Modify: `SFT/training_data_builder.py`
- Modify: `SFT/evaluator.py`
- Test: `tests/test_progress_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_prepare_dataset_uses_progress_wrapper():
    prepare_dataset([...], version_id="v1", source_name="x")
    iter_progress.assert_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_progress_integration -v`
Expected: FAIL because loops do not yet call the wrapper

- [ ] **Step 3: Write minimal implementation**

```python
for raw_line in iter_progress(lines, desc="m1", total=len(lines)):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_progress_integration -v`
Expected: PASS


### Task 3: M3 stage progress

**Files:**
- Modify: `SFT/trainer.py`
- Test: `tests/test_progress_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_trainer_stage_progress_updates_expected_steps():
    tracker = StageProgress(total=5, desc="m3")
    tracker.advance("done")
    assert tracker.n == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_progress_integration -v`
Expected: FAIL because the trainer does not use stage progress

- [ ] **Step 3: Write minimal implementation**

```python
with stage_progress(total=5, desc="m3") as progress:
    progress.advance("resolved base model")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_progress_integration -v`
Expected: PASS


### Task 4: Verification

**Files:**
- Modify: `requirements.txt`
- Test: `tests/test_progress_integration.py`

- [ ] **Step 1: Run focused tests**

Run: `python -m unittest tests.test_progress_integration tests.test_inference_backends -v`
Expected: PASS

- [ ] **Step 2: Run syntax verification**

Run: `python -m compileall SFT tests`
Expected: exit 0
