"""
Diet-plan PoC: a tiny lookup program compiled into an exact ReLU net vs the same
behaviour learned from examples.

Program:
    plan(meal, pin, filler) = NONVEG[meal] if pin == PIN else VEG[meal]
    (filler words like "please"/"today" must not change the answer)

Checks, over every meal x all 10,000 PINs:
  - right plan with wrong PIN (veg) / right PIN (non-veg)
  - false unlocks: wrong PINs that still get the non-veg plan (near-miss = 3 of 4 digits right)
  - consistency: does adding filler words change the answer?
"""
import itertools
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
MEALS = ["breakfast", "lunch", "dinner", "out_of_station"]
FILLER = ["please", "today", "quick", "for_me"]
VEG = ["Poha + fruit", "Dal, rice, sabzi", "Paneer + roti", "Veg thali / sandwich"]
NONVEG = ["Eggs + toast", "Chicken curry + rice", "Grilled fish + salad", "Chicken wrap"]
PLANS = VEG + NONVEG  # class id: meal for veg, 4 + meal for non-veg
PIN = (7, 3, 9, 1)
H = 64
SEEDS = [0, 1, 2, 3, 4]
ALL_PINS = torch.tensor(list(itertools.product(range(10), repeat=4)), dtype=torch.float)  # 10,000


def target(meal, pins):
    unlock = (pins == torch.tensor(PIN, dtype=torch.float)).all(1)
    return meal + 4 * unlock.long()


def encode(meal, pins, filler, onehot_pin):
    n = len(pins)
    m = F.one_hot(meal, 4).float()
    p = F.one_hot(pins.long(), 10).flatten(1).float() if onehot_pin else pins
    return torch.cat([m, filler, p], 1)


def mlp(n_in):
    return nn.Sequential(nn.Linear(n_in, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 8))


def compiled():
    """Raw-digit input. Layer 1: meal pass-through + per-digit equality triangles
    [d == k] = relu(d-k+1) - 2 relu(d-k) + relu(d-k-1). Layer 2: meal_m, and
    nonveg_m = relu(x_m + sum_eq - 4) (meal AND all four digits right).
    Logits: veg_m = meal_m, nonveg_m = 2 * nonveg_m. Filler gets zero weight."""
    net = mlp(4 + 4 + 4)
    for p in net.parameters():
        nn.init.zeros_(p)
    L1, L2, L3 = net[0], net[2], net[4]
    for m in range(4):
        L1.weight.data[m, m] = 1.0
    tri = []  # (unit, coefficient) pairs whose weighted sum is sum_eq
    u = 4
    for d, k in enumerate(PIN):
        col = 8 + d
        for off, c in [(-k + 1, 1.0), (-k, -2.0), (-k - 1, 1.0)]:
            L1.weight.data[u, col] = 1.0; L1.bias.data[u] = off
            tri.append((u, c)); u += 1
    for m in range(4):
        L2.weight.data[m, m] = 1.0  # meal_m
        j = 4 + m  # nonveg_m
        L2.weight.data[j, m] = 1.0
        for unit, c in tri:
            L2.weight.data[j, unit] = c
        L2.bias.data[j] = -4.0
        L3.weight.data[m, m] = 1.0
        L3.weight.data[4 + m, j] = 2.0
    return net


def sample(n, g, p_correct):
    meal = torch.randint(0, 4, (n,), generator=g)
    pins = torch.randint(0, 10, (n, 4), generator=g).float()
    hit = torch.rand(n, generator=g) < p_correct
    pins[hit] = torch.tensor(PIN, dtype=torch.float)
    filler = (torch.rand(n, 4, generator=g) < 0.3).float()
    return meal, pins, filler


def evaluate(net, onehot_pin):
    out = {}
    with torch.no_grad():
        meal = torch.arange(4).repeat_interleave(len(ALL_PINS))
        pins = ALL_PINS.repeat(4, 1)
        y = target(meal, pins)
        none = torch.zeros(len(pins), 4)
        pred = net(encode(meal, pins, none, onehot_pin)).argmax(1)
        right = (pins == torch.tensor(PIN, dtype=torch.float)).all(1)
        near = (pins == torch.tensor(PIN, dtype=torch.float)).sum(1) == 3
        out["veg_correct_wrong_pin"] = (pred[~right] == y[~right]).float().mean().item()
        out["nonveg_correct_right_pin"] = (pred[right] == y[right]).float().mean().item()
        out["false_unlocks_of_39996"] = int((pred[~right] >= 4).sum())
        out["near_miss_false_unlock_rate"] = (pred[near] >= 4).float().mean().item()
        out["wrong_meal_rate"] = ((pred % 4) != meal).float().mean().item()
        changed = torch.zeros(len(pins), dtype=torch.bool)
        for f in itertools.product([0.0, 1.0], repeat=4):
            fill = torch.tensor(f).repeat(len(pins), 1)
            changed |= net(encode(meal, pins, fill, onehot_pin)).argmax(1) != pred
        out["answer_changes_with_filler_rate"] = changed.float().mean().item()
    return out


def learned(onehot_pin, seed, n_train=20000, steps=3000, p_correct=0.2):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    meal, pins, filler = sample(n_train, g, p_correct)
    x, y = encode(meal, pins, filler, onehot_pin), target(meal, pins)
    net = mlp(x.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(steps):
        i = torch.randint(0, n_train, (512,), generator=g)
        loss = F.cross_entropy(net(x[i]), y[i])
        opt.zero_grad(); loss.backward(); opt.step()
    return net


if __name__ == "__main__":
    res = {"compiled": evaluate(compiled(), False)}
    print("compiled", json.dumps(res["compiled"]), flush=True)
    for onehot in [False, True]:
        rows = [evaluate(learned(onehot, s), onehot) for s in SEEDS]
        key = "learned_" + ("onehot_pin" if onehot else "raw_digits")
        res[key] = {"per_seed": rows,
                    "mean": {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}}
        print(key, json.dumps(res[key]["mean"]), flush=True)
    with open("results_diet.json", "w") as fh:
        json.dump(res, fh, indent=1)
