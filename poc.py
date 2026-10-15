"""
Handwired-style PoC: compile tiny integer programs into exact ReLU MLPs (no training),
then ask whether the compiled nets help *train* toy models on basic maths.

Inputs a, b are integers in [0, N). Every op is compiled into the same shape,
2 -> H -> H -> 1 with ReLU, so compiled and trained nets are directly comparable.

Key trick (exact on integers): step_k(x) = 1[x >= k] = relu(x-k+1) - relu(x-k).

E1  compiled exactness (in range + out of range)
E2  same architecture trained from scratch (70% of pairs) -> held-out + extrapolation
E3  compiled net as init for a *new* nearby op, 30% of pairs, vs random init
E4  differentiability: input gradients, and gradient-based selection among compiled ops
"""
import json
import sys

import torch
import torch.nn as nn

torch.set_num_threads(2)
N, H = 16, 64
SEEDS = [0, 1, 2, 3, 4]


def mlp():
    return nn.Sequential(nn.Linear(2, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 1))


class Builder:
    """Hand-sets weights of a 2-H-H-1 ReLU MLP. Layer-1 units are affine in (a, b);
    layer-2 units are affine in layer-1 activations; output is affine in layer 2."""

    def __init__(self):
        self.net = mlp()
        for p in self.net.parameters():
            nn.init.zeros_(p)
        self.n1 = self.n2 = 0

    def l1(self, wa, wb, bias):
        i = self.n1; self.n1 += 1
        L = self.net[0]; L.weight.data[i] = torch.tensor([wa, wb], dtype=torch.float); L.bias.data[i] = bias
        return i

    def l2(self, coeffs, bias):  # coeffs: {l1_index: weight}
        j = self.n2; self.n2 += 1
        L = self.net[2]
        for i, w in coeffs.items():
            L.weight.data[j, i] = w
        L.bias.data[j] = bias
        return j

    def out(self, coeffs, bias=0.0):
        L = self.net[4]
        for j, w in coeffs.items():
            L.weight.data[0, j] = w
        L.bias.data[0] = bias
        assert self.n1 <= H and self.n2 <= H
        return self.net

    # helpers -----------------------------------------------------------
    def pass2(self, i):  # copy a nonnegative layer-1 unit to layer 2
        return self.l2({i: 1.0}, 0.0)

    def step_units(self, wa, wb, k):  # two layer-1 units whose difference is 1[wa*a+wb*b >= k]
        return self.l1(wa, wb, -k + 1.0), self.l1(wa, wb, -float(k))


def c_add():
    B = Builder(); a = B.l1(1, 0, 0); b = B.l1(0, 1, 0)
    return B.out({B.pass2(a): 1, B.pass2(b): 1})


def c_sub():
    B = Builder(); a = B.l1(1, 0, 0); b = B.l1(0, 1, 0)
    return B.out({B.pass2(a): 1, B.pass2(b): -1})


def c_max():  # b + relu(a - b)
    B = Builder(); d = B.l1(1, -1, 0); b = B.l1(0, 1, 0)
    return B.out({B.pass2(d): 1, B.pass2(b): 1})


def c_lt():  # 1[a < b] = relu(b-a) - relu(b-a-1)
    B = Builder(); u = B.l1(-1, 1, 0); v = B.l1(-1, 1, -1)
    return B.out({B.pass2(u): 1, B.pass2(v): -1})


def c_mul():  # a*b = sum_{k=1}^{N-1} [a>=k] * b ; [a>=k]*b = relu(b - N*(1-[a>=k]))
    B = Builder(); b = B.l1(0, 1, 0); outs = {}
    for k in range(1, N):
        u, v = B.step_units(1, 0, k)
        g = B.l2({b: 1.0, u: float(N), v: -float(N)}, -float(N))
        outs[g] = 1.0
    return B.out(outs)


def c_mod(p=5, wa=1, wb=0, maxval=N - 1):  # x mod p = x - p * sum_m [x >= m p], x = wa*a + wb*b
    B = Builder(); x = B.l1(wa, wb, 0); outs = {B.pass2(x): 1.0}
    for m in range(1, maxval // p + 1):
        u, v = B.step_units(wa, wb, m * p)
        outs[B.pass2(u)] = -float(p)  # -p * (u - v) = -p * [x >= m p]
        outs[B.pass2(v)] = float(p)
    return B.out(outs)


OPS = {
    "add": (c_add, lambda a, b: a + b),
    "sub": (c_sub, lambda a, b: a - b),
    "max": (c_max, lambda a, b: torch.maximum(a, b)),
    "lt": (c_lt, lambda a, b: (a < b).float()),
    "mul": (c_mul, lambda a, b: a * b),
    "mod5": (c_mod, lambda a, b: torch.remainder(a, 5)),
}


def grid(lo, hi):
    r = torch.arange(lo, hi, dtype=torch.float)
    A, Bm = torch.meshgrid(r, r, indexing="ij")
    return torch.stack([A.flatten(), Bm.flatten()], 1)


X = grid(0, N)
X_OOR = grid(N, N + 8)  # out of range: a, b in [16, 24)


def acc(net, x, f):
    with torch.no_grad():
        y = net(x).squeeze(1)
    return (torch.round(y) == f(x[:, 0], x[:, 1])).float().mean().item()


def train(net, x, y, steps, lr, xte=None, yte_f=None, log_every=0):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    first_full = None
    for s in range(1, steps + 1):
        loss = ((net(x).squeeze(1) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if first_full is None and s % 25 == 0:
            with torch.no_grad():
                if (torch.round(net(x).squeeze(1)) == y).all():
                    first_full = s
    return first_full


def split(frac, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    perm = torch.randperm(len(X), generator=g)
    k = int(frac * len(X))
    return X[perm[:k]], X[perm[k:]]


def e1():
    res = {}
    for name, (comp, f) in OPS.items():
        net = comp()
        res[name] = {"in_range_acc": acc(net, X, f), "out_of_range_acc": acc(net, X_OOR, f),
                     "active_l1_units": int((net[0].weight.abs().sum(1) > 0).sum()),
                     "active_l2_units": int((net[2].weight.abs().sum(1) > 0).sum())}
    return res


def e2(steps=6000):
    res = {}
    for name, (_, f) in OPS.items():
        rows = []
        for s in SEEDS:
            torch.manual_seed(s)
            xtr, xte = split(0.7, s)
            net = mlp()
            train(net, xtr, f(xtr[:, 0], xtr[:, 1]), steps, 1e-3)
            rows.append((acc(net, xtr, f), acc(net, xte, f), acc(net, X_OOR, f)))
        t = torch.tensor(rows)
        res[name] = {"train_acc": t[:, 0].mean().item(), "heldout_acc": t[:, 1].mean().item(),
                     "heldout_min": t[:, 1].min().item(), "out_of_range_acc": t[:, 2].mean().item()}
        print("E2", name, res[name], flush=True)
    return res


# new targets, each paired with the compiled op it is "near"
TRANSFER = {
    "a*(b+1) <- mul": (c_mul, lambda a, b: a * (b + 1)),
    "a*b-b <- mul": (c_mul, lambda a, b: a * b - b),
    "(a+b) mod 5 <- mod5(a)": (c_mod, lambda a, b: torch.remainder(a + b, 5)),
    "2*max <- max": (c_max, lambda a, b: 2 * torch.maximum(a, b)),
    "a*b <- add (far source)": (c_add, lambda a, b: a * b),
}


def e3(steps=4000, frac=0.3):
    res = {}
    for name, (comp, f) in TRANSFER.items():
        out = {}
        out["zero_shot_acc"] = acc(comp(), X, f)
        for init in ["compiled", "random"]:
            for lr in [1e-4, 1e-3]:
                rows = []
                for s in SEEDS:
                    torch.manual_seed(s)
                    xtr, xte = split(frac, s)
                    net = comp() if init == "compiled" else mlp()
                    if init == "compiled":  # small noise so it is not a trivially frozen copy
                        with torch.no_grad():
                            for p in net.parameters():
                                p.add_(1e-3 * torch.randn_like(p))
                    ff = train(net, xtr, f(xtr[:, 0], xtr[:, 1]), steps, lr)
                    rows.append((acc(net, xte, f), acc(net, X_OOR, f), ff if ff else steps + 1))
                t = torch.tensor(rows, dtype=torch.float)
                out[f"{init}_lr{lr:g}"] = {"heldout_acc": t[:, 0].mean().item(), "heldout_min": t[:, 0].min().item(),
                                           "oor_acc": t[:, 1].mean().item(),
                                           "steps_to_100pct_train_median": t[:, 2].median().item()}
        res[name] = out
        print("E3", name, json.dumps(out), flush=True)
    return res


def e4():
    out = {}
    # (a) input gradients of compiled mul vs the true partial derivative b
    net = c_mul()
    for label, off in [("integer_points", 0.0), ("half_points", 0.5)]:
        x = (X + torch.tensor([off, 0.0])).clone().requires_grad_(True)
        net(x).sum().backward()
        g = x.grad[:, 0]
        true = X[:, 1]
        out[f"mul_dda_{label}"] = {"frac_zero_grad": (g == 0).float().mean().item(),
                                   "frac_equal_true_b": (torch.isclose(g, true)).float().mean().item(),
                                   "mean_abs_err_vs_b": (g - true).abs().mean().item()}
    # (b) differentiable program selection: out = sum softmax(w)_i op_i(a,b), ops frozen
    nets = [OPS[k][0]() for k in OPS]
    for n_ in nets:
        for p in n_.parameters():
            p.requires_grad_(False)
    with torch.no_grad():
        feats = torch.stack([n_(X).squeeze(1) for n_ in nets], 1)
    sel = {}
    for target in ["mul", "max", "mod5"]:
        f = OPS[target][1]
        y = f(X[:, 0], X[:, 1])
        torch.manual_seed(0)
        w = torch.zeros(len(nets), requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.05)
        for _ in range(500):
            loss = ((feats @ torch.softmax(w, 0) - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        pw = torch.softmax(w, 0).detach()
        sel[target] = {"picked": list(OPS)[int(pw.argmax())], "weight": round(pw.max().item(), 4)}
    out["program_selection"] = sel
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["e1", "e2", "e3", "e4"]
    results = {}
    for w in which:
        results[w] = globals()[w]()
        print(w, json.dumps(results[w], indent=1), flush=True)
    with open(f"results_{'_'.join(which)}.json", "w") as fh:
        json.dump(results, fh, indent=1)
