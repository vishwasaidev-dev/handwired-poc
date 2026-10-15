# Hand-wired vs learned neural networks

Small experiments on networks that are compiled directly from programs, compared with the same networks learned from examples.

## Credit

The idea comes from **Cheng Lou** ([@_chenglou](https://x.com/_chenglou)), who [introduced Handwired](https://x.com/_chenglou/status/2110557373651931267) on 15 October 2026. Handwired is a toy project that turns ordinary JavaScript into a neural network with no training, exact outputs and full differentiability, running in a browser.

This repo does not contain or use Handwired's code. It is an independent re-creation of the idea in Python and PyTorch, plus follow-up experiments. Any mistakes are ours.

## The trick

On whole-number inputs, two ReLU neurons make an exact switch:

```
relu(x - k + 1) - relu(x - k) = 1 if x >= k else 0
```

Combining these switches gives comparisons, lookups, multiplication over a fixed range, and exact equality checks. Every network here has the same shape: inputs → 64 → 64 → outputs, with ReLU activations.

## Results

### 1. Basic maths (`poc.py`)

Inputs are integers a and b from 0 to 15. Learned networks train on 70% of the pairs and are scored on the other 30% (5 seeds).

| Operation | Hand-wired, in range | Hand-wired, inputs 16–23 | Learned, unseen pairs | Learned, inputs 16–23 |
| --- | --- | --- | --- | --- |
| add, subtract, max, less-than | 100% | 100% | 100% | 100% |
| multiply | 100% | 0% | 49% | 0% |
| a mod 5 | 100% | 50% | 50% | 9% |

**Hand-wired as a head start.** A hand-wired net is trained on a nearby op using only 30% of the pairs. Scores are on unseen pairs, over two learning rates.

| New op ← start | Hand-wired start | Random start |
| --- | --- | --- |
| a·(b+1) ← multiply | 100% | 8–26% |
| a·b − b ← multiply | 96% | 7–29% |
| 2·max ← max | 91–100% | 96–100% |
| (a+b) mod 5 ← a mod 5 | 19% | 18% |
| a·b ← add | 4–11% | 7–25% |

The head start helps only for small changes to the original op. For the hand-wired multiplier, d(a·b)/da matched the true value b at only 6% of integer points.

### 2. Diet planner with a PIN (`diet.py`)

Asking for a meal (breakfast, lunch, dinner or out of station) returns its veg plan. With PIN 7391 it returns the non-veg plan instead. Filler words must not change the answer. Every meal is tested against all 10,000 PINs.

| Check | Hand-wired | Learned (10 nets) |
| --- | --- | --- |
| Veg plan with a wrong PIN | 100% | 99.98% |
| Non-veg plan with the right PIN | 100% | 100% |
| Wrong PINs that unlocked (of 39,996) | 0 | 0–17 per net |
| One-digit-off PINs that unlocked | 0% | ~4% |

### 3. Drift when training a hand-wired net (`drift.py`, `drift_fix.py`)

Fine-tuning the perfect hand-wired planner, even on examples of its own rule, made ~94% of one-digit-off PINs unlock within 300 steps (lr 1e-4). The cause: near-miss PINs sit exactly on a ReLU threshold, so they send back no gradient to resist the shift.

These numbers come from teaching it a small rule change (3 seeds, 3,000 steps):

| Approach | Learned the change | Wrong PINs that unlocked | One-digit-off PINs that unlocked |
| --- | --- | --- | --- |
| Train everything | 99.8% | 69–81 | 46–49% |
| Hand-wire with a safety margin | 99.9% | 35–44 | 23–28% |
| Freeze hand-wired layers, train the last layer only | 100% | 0 | 0% |

## Conclusions

1. Hand-wired networks are exact, but only within the range they were built for.
2. Learned networks get simple rules right, and complex or exact rules nearly right.
3. A hand-wired start is a strong head start for nearby tasks. It is no help for unrelated tasks, and training cannot add parts the network lacks.
4. "Differentiable" does not mean "correct derivatives".
5. Training smudges hand-wired parts unless you freeze them.

## Caveats

- These are toy experiments that take minutes on a CPU, with 3–5 seeds per setting.
- The from-scratch maths baseline is untuned: inputs are unscaled and there is no learning-rate sweep.
- This is a re-creation of the idea, not Handwired itself.

## Run

Requires Python 3 and PyTorch (CPU is fine).

```
python poc.py          # maths: e1 e2 e3 e4 (or pass a subset, e.g. python poc.py e1)
python diet.py         # diet planner
python drift.py        # drift
python drift_fix.py    # fixes (imports diet.py and drift.py)
```

Each script prints its results and writes a `results_*.json`. The JSON files from our runs are included.
