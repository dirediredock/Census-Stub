import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.sans-serif": "Consolas",
        "font.weight": "bold",
        "font.size": 16,
    }
)

color_highlight = "tab:red"

figure_size = 6


def get_XY_from_CSV(file):
    df = pd.read_csv(file)
    df = df.sort_values(by="byte")
    val_X = list(df["byte"])
    val_Y = list(df["frequency"])
    return val_X, val_Y


val_X_graph6, val_Y_graph6 = get_XY_from_CSV("TGC_bytes/graph6.csv")
val_X_edgelist, val_Y_edgelist = get_XY_from_CSV("TGC_bytes/edgelist.csv")
val_X_diameter, val_Y_diameter = get_XY_from_CSV("TGC_bytes/diameter.csv")
val_X_DD, val_Y_DD = get_XY_from_CSV("TGC_bytes/degree_distribution.csv")
val_X_CN, val_Y_CN = get_XY_from_CSV("TGC_bytes/CN.csv")
val_X_CE, val_Y_CE = get_XY_from_CSV("TGC_bytes/CE.csv")
val_X_CS, val_Y_CS = get_XY_from_CSV("TGC_bytes/CS.csv")
val_X_BN, val_Y_BN = get_XY_from_CSV("TGC_bytes/BN.csv")
val_X_BE, val_Y_BE = get_XY_from_CSV("TGC_bytes/BE.csv")
val_X_BS, val_Y_BS = get_XY_from_CSV("TGC_bytes/BS.csv")

val_X = [
    val_X_diameter,
    val_X_graph6,
    val_X_DD,
    val_X_BN,
    val_X_CN,
    val_X_CE,
    val_X_CS,
    val_X_edgelist,
    val_X_BE,
    val_X_BS,
]

val_Y = [
    val_Y_diameter,
    val_Y_graph6,
    val_Y_DD,
    val_Y_BN,
    val_Y_CN,
    val_Y_CE,
    val_Y_CS,
    val_Y_edgelist,
    val_Y_BE,
    val_Y_BS,
]

val_name = [
    " Diameter",
    " Graph6",
    " Degree Sequence",
    " BMatrix-Node",
    " Census-Node",
    " Census-Edge",
    " Census-Stub",
    " Edgelist",
    " BMatrix-Edge",
    " BMatrix-Stub",
]

###############################################################################

print()

fig = plt.figure(figsize=(figure_size, figure_size))
ax = fig.add_subplot(111)

global_min_X = np.inf
global_max_X = 0

global_max_Y = 0

for idx in range(0, 10):
    global_min_X = int(np.min([global_min_X, np.min(val_X[idx])]))
    global_max_X = int(np.max([global_max_X, np.max(val_X[idx])]))

    if idx in [0, 1, 2, 7]:
        highlight = "gray"
    elif idx in [6]:
        highlight = "red"
    else:
        highlight = "black"

    max_Y = np.max(val_Y[idx])
    for bar in range(0, len(val_X[idx])):
        global_max_Y = int(
            np.max([global_max_Y, ((val_Y[idx][bar] / max_Y) * 0.85) + idx]),
        )
        ax.plot(
            [val_X[idx][bar], val_X[idx][bar]],
            [0 + idx, ((val_Y[idx][bar] / max_Y) * 0.85) + idx],
            "-",
            c=highlight,
            linewidth=1.1,
            solid_capstyle="round",
        )
        ax.scatter(
            [val_X[idx][bar], val_X[idx][bar]],
            [0 + idx, ((val_Y[idx][bar] / max_Y) * 0.85) + idx],
            c=highlight,
            s=1,
            edgecolors="none",
        )

    # ax.annotate(
    #     val_name[idx],
    #     xy=(np.max(val_X[idx]) + 5, idx - 0.045),
    #     ha="left",
    #     va="bottom",
    #     color=highlight,
    #     fontsize=12,
    # )

    all_bytes = list(
        itertools.chain.from_iterable(
            itertools.repeat(val, freq)
            for val, freq in zip(
                val_X[idx],
                val_Y[idx],
            )
        )
    )

    print("\n", val_name[idx])
    print("\t\tmin:", np.min(all_bytes))
    print("\t\tmean:", round(np.mean(all_bytes), 3))
    print("\t\tmax:", np.max(all_bytes))


ax.set_xlim(global_min_X - 3, global_max_X + 100)
ax.set_ylim(-0.3, global_max_Y + 1)
ax.set_xticks(np.linspace(global_min_X, global_max_X, 6))

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_yticks([])

# ax.set_xlabel("Storage Size (bytes)")

plt.tight_layout()

# plt.show(block=False)

plt.savefig(f"Collider_03.png", format="png", dpi=800)
plt.close()

###############################################################################
