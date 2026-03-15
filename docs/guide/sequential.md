# Sequential Testing

The peeking problem: if you check results daily and stop when you see significance, your false positive rate inflates far beyond 5%. Sequential testing methods give you always-valid inference that stays correct no matter when you look.

## mSPRT

The mixture Sequential Probability Ratio Test provides always-valid p-values via a mixture likelihood ratio.

```python
from splita import mSPRT
import numpy as np

test = mSPRT(metric='continuous', alpha=0.05)

# Day 1: first batch of data
rng = np.random.default_rng(1)
ctrl_day1 = rng.normal(10, 2, size=100)
trt_day1 = rng.normal(10.5, 2, size=100)
state = test.update(ctrl_day1, trt_day1)
print(state.should_stop)          # False
print(state.always_valid_pvalue)  # still high

# Day 2: more data arrives (incremental update)
ctrl_day2 = np.random.default_rng(3).normal(10, 2, size=200)
trt_day2 = np.random.default_rng(4).normal(10.5, 2, size=200)
state = test.update(ctrl_day2, trt_day2)
print(state.should_stop)          # may now be True
print(state.always_valid_pvalue)
```

### For conversion metrics

```python
test = mSPRT(metric='conversion', alpha=0.05)

ctrl = np.random.default_rng(1).binomial(1, 0.10, size=500)
trt = np.random.default_rng(2).binomial(1, 0.12, size=500)
state = test.update(ctrl, trt)
```

## GroupSequential

Classical alpha-spending boundaries (O'Brien-Fleming, Pocock, Kim-DeMets). You pre-specify the number of interim looks.

```python
from splita import GroupSequential
import numpy as np

gs = GroupSequential(n_looks=4, alpha=0.05, spending='obf')

# Look 1 (25% of data)
rng = np.random.default_rng(42)
ctrl = rng.normal(10, 2, 250)
trt = rng.normal(10.8, 2, 250)
result = gs.update(ctrl, trt)
print(result.boundary)    # critical value at this look
print(result.should_stop) # usually False at early looks with OBF

# Look 2 (50% of data)
ctrl2 = rng.normal(10, 2, 250)
trt2 = rng.normal(10.8, 2, 250)
result = gs.update(ctrl2, trt2)
```

### Spending functions

| Function | Behavior |
|----------|----------|
| `'obf'` | O'Brien-Fleming: conservative early, aggressive late. Most common. |
| `'pocock'` | Equal spending at each look. Easier to stop early. |
| `'kim-demets'` | Parameterized power family. Flexible. |

## ConfidenceSequence

Time-uniform confidence sequences that are valid at every sample size simultaneously.

```python
from splita import ConfidenceSequence
import numpy as np

cs = ConfidenceSequence(alpha=0.05)

rng = np.random.default_rng(42)

# Update with each batch
for day in range(1, 8):
    ctrl = rng.normal(10, 2, 100)
    trt = rng.normal(10.5, 2, 100)
    state = cs.update(ctrl, trt)
    print(f"Day {day}: CI = [{state.lower:.3f}, {state.upper:.3f}], "
          f"significant = {state.significant}")
```

!!! tip
    Confidence sequences are wider than fixed-sample CIs at any single time point, but they are valid at all time points. This is the price of continuous monitoring.

## EValue

E-value sequential testing. An alternative to p-values that composes under optional continuation.

```python
from splita import EValue
import numpy as np

ev = EValue(alpha=0.05)

rng = np.random.default_rng(42)
ctrl = rng.normal(10, 2, 500)
trt = rng.normal(10.5, 2, 500)
state = ev.update(ctrl, trt)
print(state.e_value)       # evidence against H0
print(state.should_stop)   # True if e_value > 1/alpha
```

## EProcess

Safe testing with e-processes (GRAPA, universal inference).

```python
from splita import EProcess
import numpy as np

ep = EProcess(alpha=0.05)

rng = np.random.default_rng(42)
for _ in range(5):
    ctrl = rng.normal(10, 2, 100)
    trt = rng.normal(10.5, 2, 100)
    state = ep.update(ctrl, trt)
    if state.should_stop:
        print("Rejected H0")
        break
```

## SampleSizeReestimation

Mid-experiment sample size adjustment based on conditional power:

```python
from splita import SampleSizeReestimation
import numpy as np

reest = SampleSizeReestimation(
    original_n=1000,
    alpha=0.05,
    power=0.80,
)

# At the interim look (50% of planned sample)
rng = np.random.default_rng(42)
ctrl = rng.normal(10, 2, 500)
trt = rng.normal(10.3, 2, 500)
result = reest.update(ctrl, trt)
print(result.conditional_power)
print(result.recommended_n)
```

## YEASTSequentialTest

Tuning-free sequential test from Meta, based on Levy's inequality. No mixing distribution to choose.

```python
from splita import YEASTSequentialTest
import numpy as np

yeast = YEASTSequentialTest(alpha=0.05)

rng = np.random.default_rng(42)
ctrl = rng.normal(10, 2, 500)
trt = rng.normal(10.5, 2, 500)
state = yeast.update(ctrl, trt)
print(state.should_stop)
print(state.statistic)
```

## When to use which

| Method | Best for |
|--------|----------|
| `mSPRT` | General-purpose continuous monitoring |
| `GroupSequential` | Pre-planned interim analyses (regulatory) |
| `ConfidenceSequence` | When you need a confidence interval at every point |
| `EValue` / `EProcess` | When you want evidence that composes across experiments |
| `SampleSizeReestimation` | When initial power assumptions may be wrong |
| `YEASTSequentialTest` | When you don't want to tune hyperparameters |
