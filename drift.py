"""
Does training make a hand-wired net drift?

Start from diet.compiled() (exact on all 40,000 meal x PIN cases) and fine-tune:
  A  on data from the SAME rule it already implements
  B  on a small rule update: out_of_station + wrong PIN -> lunch veg plan (class 1)
Measure, over all 40,000 cases, what still matches the intended rule,
split into the part we meant to change and everything else.
"""
import json

import torch
import torch.nn.functional as F

import diet

SEEDS = [0, 1, 2]
CASES_MEAL = torch.arange(4).repeat_interleave(len(diet.ALL_PINS))
CASES_PIN = diet.ALL_PINS.repeat(4, 1)
RIGHT = (CASES_PIN == torch.tensor(diet.PIN, dtype=torch.float)).all(1)
NEAR = (CASES_PIN == torch.tensor(diet.PIN, dtype=torch.float)).sum(1) == 3


def rule_same(meal, pins):
    return diet.target(meal, pins)


def rule_update(meal, pins):
    y = diet.target(meal, pins).clone()
    y[(meal == 3) & (y == 3)] = 1  # out_of_station veg -> lunch veg plan
    return y


def check(net, rule):
    with torch.no_grad():
        x = diet.encode(CASES_MEAL, CASES_PIN, torch.zeros(len(CASES_PIN), 4), False)
        pred = net(x).argmax(1)
    y = rule(CASES_MEAL, CASES_PIN)
    changed_part = (CASES_MEAL == 3) & ~RIGHT if rule is rule_update else torch.zeros_like(RIGHT)
    rest = ~changed_part
    return {
        "updated_part_correct": (pred[changed_part] == y[changed_part]).float().mean().item() if changed_part.any() else None,
        "everything_else_correct": (pred[rest] == y[rest]).float().mean().item(),
        "nonveg_with_right_pin": (pred[RIGHT] == y[RIGHT]).float().mean().item(),
        "false_unlocks_of_39996": int((pred[~RIGHT] >= 4).sum()),
        "near_miss_false_unlock_rate": (pred[NEAR] >= 4).float().mean().item(),
    }


def finetune(rule, lr, steps, seed, n=20000, p_correct=0.2):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    meal, pins, filler = diet.sample(n, g, p_correct)
    x, y = diet.encode(meal, pins, filler, False), rule(meal, pins)
    net = diet.compiled()
    w0 = torch.cat([p.detach().flatten().clone() for p in net.parameters()])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        i = torch.randint(0, n, (512,), generator=g)
        loss = F.cross_entropy(net(x[i]), y[i])
        opt.zero_grad(); loss.backward(); opt.step()
    w1 = torch.cat([p.detach().flatten() for p in net.parameters()])
    out = check(net, rule)
    out["weight_change_l2"] = (w1 - w0).norm().item()
    return out


if __name__ == "__main__":
    res = {"before_training": {"same_rule": check(diet.compiled(), rule_same),
                               "updated_rule": check(diet.compiled(), rule_update)}}
    print("before", json.dumps(res["before_training"]), flush=True)
    for name, rule in [("A_same_rule", rule_same), ("B_rule_update", rule_update)]:
        for lr in [1e-4, 1e-3]:
            for steps in [300, 3000]:
                rows = [finetune(rule, lr, steps, s) for s in SEEDS]
                key = f"{name}_lr{lr:g}_steps{steps}"
                res[key] = rows
                print(key, json.dumps(rows), flush=True)
    with open("results_drift.json", "w") as fh:
        json.dump(res, fh, indent=1)
