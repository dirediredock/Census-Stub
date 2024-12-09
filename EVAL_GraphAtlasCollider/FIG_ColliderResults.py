from math import comb

import matplotlib.pyplot as plt
from numpy import nan as NaN

plt.rcParams.update(
    {
        "font.sans-serif": "Consolas",
        "font.weight": "bold",
        "font.size": 16,
    }
)

color_highlight = "tab:red"

figure_size = 6

###############################################################################

graph_order = [3, 4, 5, 6, 7, 8, 9, 10]

# EMPIRICAL

nonisomorphic = [4, 11, 34, 156, 1044, 12346, 274668, 12005168]
disconnected = [2, 5, 13, 44, 191, 1229, 13588, 288597]

ceiling_nonisomorphic = []
ceiling_connected_only = []

for i, num in enumerate(nonisomorphic):
    ceiling_nonisomorphic.append(comb(num, 2))
    connected_only = num - disconnected[i]
    ceiling_connected_only.append(comb(connected_only, 2))

# LITERATURE

oeis_A000088 = [4, 11, 34, 156, 1044, 12346, 274668, 12005168]
oeis_A001349 = [2, 6, 21, 112, 853, 11117, 261080, 11716571]

ceiling_oeis_A000088 = []
ceiling_oeis_A001349 = []

for i in range(len(oeis_A000088)):
    ceiling_oeis_A000088.append(comb(oeis_A000088[i], 2))
    ceiling_oeis_A001349.append(comb(oeis_A001349[i], 2))

# # SANITY CHECK

# print()
# print(ceiling_nonisomorphic == ceiling_oeis_A000088)
# print(ceiling_connected_only == ceiling_oeis_A001349)
# print()

# # x = [3, 4, 5, 6, 7, 8, 9, 10]
# # y = [1, 15, 210, 6216, 363378, 61788286, 34081252660, 68639012140735]

###############################################################################

Census_of_Nodes = [NaN, NaN, 1, 23, 871, 76722, 23005084, 26627947737]
Census_of_Edges = [NaN, NaN, NaN, 4, 92, 3218, 207782, 34079114]
Census_of_Stubs = [NaN, NaN, NaN, NaN, NaN, 27, 2691, 336711]

BMatrix_of_Nodes = [NaN, NaN, 1, 23, 875, 77134, 23162738, 26766491001]
BMatrix_of_Edges = [NaN, NaN, NaN, 4, 100, 3765, 251426, 41264826]
BMatrix_of_Stubs = [NaN, NaN, NaN, NaN, 1, 77, 6940, 827217]

diameter = [NaN, 6, 101, 2642, 147426, 25960154, 15361859890, 32873337922477]
degree = [NaN, NaN, 2, 75, 3048, 293364, 90277837, 101526363676]

###############################################################################

A = ceiling_connected_only

# CN_CE_CS_ = [value / A[i] * 100 for i, value in enumerate(CN_CE_CS)]
# CN_CE_xx_ = [value / A[i] * 100 for i, value in enumerate(CN_CE_xx)]
# CN_xx_CS_ = [value / A[i] * 100 for i, value in enumerate(CN_xx_CS)]
# CN_xx_xx_ = [value / A[i] * 100 for i, value in enumerate(CN_xx_xx)]
# xx_CE_CS_ = [value / A[i] * 100 for i, value in enumerate(xx_CE_CS)]
# xx_CE_xx_ = [value / A[i] * 100 for i, value in enumerate(xx_CE_xx)]
# xx_xx_CS_ = [value / A[i] * 100 for i, value in enumerate(xx_xx_CS)]
# xx_xx_xx_ = [value / A[i] * 100 for i, value in enumerate(xx_xx_xx)]

CN_ = [value / A[i] * 100 for i, value in enumerate(Census_of_Nodes)]
CE_ = [value / A[i] * 100 for i, value in enumerate(Census_of_Edges)]
CS_ = [value / A[i] * 100 for i, value in enumerate(Census_of_Stubs)]

BN_ = [value / A[i] * 100 for i, value in enumerate(BMatrix_of_Nodes)]
BE_ = [value / A[i] * 100 for i, value in enumerate(BMatrix_of_Edges)]
BS_ = [value / A[i] * 100 for i, value in enumerate(BMatrix_of_Stubs)]

diameter_ = [value / A[i] * 100 for i, value in enumerate(diameter)]
degree_ = [value / A[i] * 100 for i, value in enumerate(degree)]

###############################################################################

fig = plt.figure(figsize=(figure_size, figure_size))
ax = fig.add_subplot(111)

ax.plot(
    graph_order,
    ceiling_connected_only,
    ".-k",
    markersize=8,
    label="  Collision Ceiling",
)

ax.scatter(NaN, NaN, c="w", label=" ")

ax.plot(
    graph_order,
    diameter,
    "+:k",
    alpha=0.5,
    markersize=8,
    markeredgecolor="k",
    # markerfacecolor="white",
    label="  Diameter",
    zorder=10,
)

ax.plot(
    graph_order,
    degree,
    "x:k",
    alpha=0.5,
    markersize=6,
    markeredgecolor="k",
    # markerfacecolor="white",
    label="  Degree Distribution",
    zorder=10,
)

ax.scatter(NaN, NaN, c="w", label=" ")

ax.plot(
    graph_order,
    BMatrix_of_Nodes,
    color="k",
    marker="o",
    linestyle="dotted",
    markersize=13,
    markerfacecolor="white",
    label="  BMatrix-Node",
)
ax.plot(
    graph_order,
    Census_of_Nodes,
    ".:",
    color="k",
    markersize=9,
    label="  Census-Node",
)

ax.scatter(NaN, NaN, c="w", label=" ")

ax.plot(
    graph_order,
    BMatrix_of_Edges,
    color="k",
    marker="s",
    linestyle="dotted",
    markersize=10,
    markerfacecolor="white",
    label="  BMatrix-Edge",
)
ax.plot(
    graph_order,
    Census_of_Edges,
    color="k",
    marker="s",
    linestyle="dotted",
    markersize=5,
    markerfacecolor="k",
    label="  Census-Edge",
)

ax.scatter(NaN, NaN, c="w", label=" ")

ax.plot(
    graph_order,
    BMatrix_of_Stubs,
    color="k",
    marker="D",
    linestyle="dotted",
    markersize=10,
    markerfacecolor="white",
    label="  BMatrix-Stub",
)
ax.plot(
    graph_order,
    Census_of_Stubs,
    color="r",
    marker="D",
    linestyle="dotted",
    markersize=5,
    markerfacecolor="r",
    label="  Census-Stub",
)

ax.scatter(NaN, NaN, c="w", label=" ")

ax.set_yscale("log")

# legend = ax.legend(
#     edgecolor="none",
#     handlelength=6,
#     borderaxespad=4.2,
#     # weight="normal",
#     fontsize=10.6,
# )
# legend.get_texts()[12].set_color("r")

ax.set_xticks(range(3, 11))
# ax.set_xlim(2.3, 13)

plt.tight_layout()

# plt.show(block=False)

plt.savefig(f"Collider_01.png", format="png", dpi=800)
plt.close()

###############################################################################

fig = plt.figure(figsize=(figure_size, figure_size))
ax = fig.add_subplot(111)

ax.plot(
    graph_order,
    [100] * len(graph_order),
    ".-k",
    markersize=8,
    label="  Collision Ceiling",
)

ax.scatter(NaN, NaN, c="w", label=" ")

ax.plot(
    graph_order,
    diameter_,
    "+:k",
    alpha=0.5,
    markersize=8,
    markeredgecolor="k",
    # markerfacecolor="white",
    label="  Diameter",
    zorder=10,
)

ax.plot(
    graph_order,
    degree_,
    "x:k",
    alpha=0.5,
    markersize=6,
    markeredgecolor="k",
    # markerfacecolor="white",
    label="  Degree Distribution",
    zorder=10,
)

ax.scatter(NaN, NaN, c="w", label=" ")

ax.plot(
    graph_order,
    BN_,
    color="k",
    marker="o",
    linestyle="dotted",
    markersize=13,
    markerfacecolor="white",
    label="  BMatrix-Node",
)
ax.plot(
    graph_order,
    CN_,
    ".:",
    color="k",
    markersize=9,
    label="  Census-Node",
)

ax.scatter(NaN, NaN, c="w", label=" ")

ax.plot(
    graph_order,
    BE_,
    color="k",
    marker="s",
    linestyle="dotted",
    markersize=10,
    markerfacecolor="white",
    label="  BMatrix-Edge",
)
ax.plot(
    graph_order,
    CE_,
    color="k",
    marker="s",
    linestyle="dotted",
    markersize=5,
    markerfacecolor="k",
    label="  Census-Edge",
)

ax.scatter(NaN, NaN, c="w", label=" ")

ax.plot(
    graph_order,
    BS_,
    color="k",
    marker="D",
    linestyle="dotted",
    markersize=10,
    markerfacecolor="white",
    label="  BMatrix-Stub",
)
ax.plot(
    graph_order,
    CS_,
    color="r",
    marker="D",
    linestyle="dotted",
    markersize=5,
    markerfacecolor="r",
    label="  Census-Stub",
)

# ax.scatter(NaN, NaN, c="w", label=" ")

ax.set_yscale("log")

# legend = ax.legend(
#     edgecolor="none",
#     handlelength=5.2,
#     borderaxespad=2.5,
#     fontsize=9.6,
# )
# legend.get_texts()[12].set_color("r")

ax.set_xticks(range(3, 11))
ax.set_xlim(2.3, 13)

plt.tight_layout()

# plt.show(block=False)

plt.savefig(f"Collider_02.png", format="png", dpi=800)
plt.close()

###############################################################################

plt.show()
