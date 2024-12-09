from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from numpy import nan

plt.rcParams.update(
    {
        "font.sans-serif": "Consolas",
        "font.weight": "bold",
        "font.size": 16,
    }
)


figure_size = 6

C = [
    "#9c755f",
    "#f28e2b",
    "#4e79a7",
    # "#e15759",
    "r",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#76b7b2",
    "#ff9da7",
    "#bab0ac",
]
L = 2.7
M = 20

###############################################################################

graphs_order = [3, 4, 5, 6, 7, 8, 9, 10]
graphs_N = [2, 6, 21, 112, 853, 11117, 261080, 11716571]

CN_CE_CS = [nan, nan, nan, nan, nan, 45, 4553, 535723]
CN_CE_xx = [nan, nan, nan, 8, 118, 2706, 95143, 5655161]
CN_xx_CS = [nan, nan, nan, nan, nan, nan, nan, 503]
CN_xx_xx = [nan, nan, 2, 26, 332, 5545, 135346, 5194335]
xx_CE_CS = [nan, nan, nan, nan, nan, nan, nan, 2]
xx_CE_xx = [nan, nan, nan, nan, 2, 24, 295, 5355]
xx_xx_CS = [nan, nan, nan, nan, nan, nan, nan, 17]
xx_xx_xx = [2, 6, 19, 78, 401, 2797, 25743, 325475]

CN_CE_CS_ = [value / graphs_N[i] * 100 for i, value in enumerate(CN_CE_CS)]
CN_CE_xx_ = [value / graphs_N[i] * 100 for i, value in enumerate(CN_CE_xx)]
CN_xx_CS_ = [value / graphs_N[i] * 100 for i, value in enumerate(CN_xx_CS)]
CN_xx_xx_ = [value / graphs_N[i] * 100 for i, value in enumerate(CN_xx_xx)]
xx_CE_CS_ = [value / graphs_N[i] * 100 for i, value in enumerate(xx_CE_CS)]
xx_CE_xx_ = [value / graphs_N[i] * 100 for i, value in enumerate(xx_CE_xx)]
xx_xx_CS_ = [value / graphs_N[i] * 100 for i, value in enumerate(xx_xx_CS)]
xx_xx_xx_ = [value / graphs_N[i] * 100 for i, value in enumerate(xx_xx_xx)]

###############################################################################

fig = plt.figure(figsize=(figure_size, figure_size))
ax = fig.add_subplot(111)

ax.plot(graphs_order, [100] * 8, ".-", c=C[-1], linewidth=L, markersize=M)
ax.plot(graphs_order, xx_xx_xx_, ".-", c=C[7], linewidth=L, markersize=M)
ax.plot(graphs_order, CN_xx_xx_, ".-", c=C[5], linewidth=L, markersize=M)
ax.plot(graphs_order, CN_CE_xx_, ".-", c=C[2], linewidth=L, markersize=M)
ax.plot(graphs_order, CN_xx_CS_, ".-", c=C[1], linewidth=L, markersize=M)
ax.plot(graphs_order, xx_CE_CS_, ".-", c=C[0], linewidth=L, markersize=M)
ax.plot(graphs_order, CN_CE_CS_, ".-", c=C[6], linewidth=L, markersize=M)
ax.plot(graphs_order, xx_CE_xx_, ".-", c=C[4], linewidth=L, markersize=M)
ax.plot(graphs_order, xx_xx_CS_, ".-", c=C[3], linewidth=L, markersize=M)

ax.set_xticks(range(3, 11))
ax.set_xlim(2.3, 13)

plt.tight_layout()

plt.show(block=False)

# plt.savefig(f"Set_01.png", format="png", dpi=800)
# plt.close()

###############################################################################

fig = plt.figure(figsize=(figure_size, figure_size))
ax = fig.add_subplot(111)

ax.plot(graphs_order, [100] * 8, ".-", c=C[-1], linewidth=L, markersize=M)
ax.plot(graphs_order, xx_xx_xx_, ".-", c=C[7], linewidth=L, markersize=M)
ax.plot(graphs_order, CN_xx_xx_, ".-", c=C[5], linewidth=L, markersize=M)
ax.plot(graphs_order, CN_CE_xx_, ".-", c=C[2], linewidth=L, markersize=M)
ax.plot(graphs_order, CN_xx_CS_, ".-", c=C[1], linewidth=L, markersize=M)
ax.plot(graphs_order, xx_CE_CS_, ".-", c=C[0], linewidth=L, markersize=M)
ax.plot(graphs_order, CN_CE_CS_, ".-", c=C[6], linewidth=L, markersize=M)
ax.plot(graphs_order, xx_CE_xx_, ".-", c=C[4], linewidth=L, markersize=M)
ax.plot(graphs_order, xx_xx_CS_, ".-", c=C[3], linewidth=L, markersize=M)

ax.set_yscale("log")

ax.set_xticks(range(3, 11))
ax.set_xlim(2.3, 13)

plt.tight_layout()

plt.show(block=False)

# plt.savefig(f"Set_02.png", format="png", dpi=800)
# plt.close()

###############################################################################

plt.show()
