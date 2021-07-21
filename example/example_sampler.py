import sys

import numpy as np

sys.path.insert(0, "./")

import matplotlib.pyplot as plt
from bayes_optim import DiscreteSpace, IntegerSpace, RealSpace
from bayes_optim.samplers import SCMC
from scipy.stats import norm

dim = 2
search_space = RealSpace([0, 1]) * dim
h = lambda x: np.abs(np.sum(x) - 1)
tol = 1e-2
sampler = SCMC(
    search_space,
    [
        h,
    ],
    tol=tol,
)

for i in range(20):
    X = sampler.sample(5)
    print(X)
    assert np.all([np.isclose(h(x), 0, atol=tol) for x in X])


dim = 2
search_space = (
    RealSpace([-5, 5]) * dim + DiscreteSpace(["A", "B", "C", "D"]) + IntegerSpace([1, 10])
)
sampler = SCMC(
    search_space,
    [
        lambda x: np.sum(x[:2] ** 2) - 1,
        lambda x: 0.25 - np.sum(x[:2] ** 2),
        lambda x: bool(x[2] not in ["A", "B"]),
        lambda x: x[3] - 5.1,
    ],
    # target_dist=lambda x: norm.pdf(x[0], scale=3) * norm.pdf(x[1]),
    tol=tol,
)
X = sampler.sample(100)
assert np.all([x[2] in ["A", "B"] for x in X])

plt.plot(X[:, 0], X[:, 1], "r.")
circle1 = plt.Circle((0, 0), 1, color="k", fill=False)
circle2 = plt.Circle((0, 0), 0.5, color="k", fill=False)

ax = plt.gca()
ax.add_patch(circle1)
ax.add_patch(circle2)
plt.show()
