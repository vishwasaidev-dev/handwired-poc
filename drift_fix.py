"""Two fixes for drift in the hand-wired diet net, tested on rule update B (lr 1e-4, 300 and 3000 steps):
  margin  - rebuild with slack: nonveg unit bias -4.5 (near-miss sits at -0.5, not on the ReLU hinge), output weight 4
  freeze  - train only the output layer; the hand-wired PIN/meal detectors stay fixed
"""
import json
import torch
import torch.nn.functional as F
import diet, drift


def compiled_margin():
    net = diet.compiled()
    for m in range(4):
        net[2].bias.data[4 + m] = -4.5
        net[4].weight.data[4 + m, 4 + m] = 4.0
    return net


def run(build, freeze, steps, seed, lr=1e-4, n=20000):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    meal, pins, filler = diet.sample(n, g, 0.2)
    x, y = diet.encode(meal, pins, filler, False), drift.rule_update(meal, pins)
    net = build()
    params = net[4].parameters() if freeze else net.parameters()
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        i = torch.randint(0, n, (512,), generator=g)
        loss = F.cross_entropy(net(x[i]), y[i]); opt.zero_grad(); loss.backward(); opt.step()
    return drift.check(net, drift.rule_update)


print("margin before training", json.dumps(drift.check(compiled_margin(), drift.rule_same)))
res = {}
for name, build, freeze in [("margin", compiled_margin, False), ("freeze", diet.compiled, True), ("margin+freeze", compiled_margin, True)]:
    for steps in [300, 3000]:
        rows = [run(build, freeze, steps, s) for s in drift.SEEDS]
        res[f"{name}_steps{steps}"] = rows
        print(name, steps, json.dumps({k: [round(r[k], 4) if isinstance(r[k], float) else r[k] for r in rows] for k in rows[0]}), flush=True)
with open("results_drift_fix.json", "w") as fh:
    json.dump(res, fh, indent=1)
