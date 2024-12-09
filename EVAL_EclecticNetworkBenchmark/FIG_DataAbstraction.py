from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np

figure_size = 5

plt.rcParams.update(
    {
        "font.sans-serif": "Fira Code",
        "font.weight": "bold",
        "font.size": 10,
    }
)

new_tableau_10 = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]


def BFS_Census(G):
    D = nx.to_dict_of_dicts(G)
    Census_of_Nodes = []
    Census_of_Edges = []
    Census_of_Stubs = []
    for source_node in D.keys():
        Q = [source_node]
        visited_nodes = set(Q)
        visited_edges = set()
        visited_stubs = set()
        vector_of_node_degrees = []
        vector_of_edge_degrees = []
        vector_of_stub_degrees = []
        while len(Q) > 0:
            node_degree = 0
            edge_degree = 0
            stub_degree = 0
            current_stubs = set()
            upcoming_nodes = []
            for node in Q:
                neighbors = D[node].keys()
                for neighbor in neighbors:
                    if neighbor not in visited_nodes:
                        upcoming_nodes.append(neighbor)
                        node_degree += 1
                    visited_nodes.add(neighbor)
                    edge = (min(node, neighbor), max(node, neighbor))
                    if edge not in visited_edges:
                        edge_degree += 1
                    visited_edges.add(edge)
                    stub = (node, neighbor)
                    if stub not in visited_stubs:
                        stub_degree += 1
                    current_stubs.add(stub)
                    visited_stubs.add(stub)
            for stub in current_stubs:
                visited_stubs.add((stub[1], stub[0]))
            vector_of_node_degrees.append(node_degree)
            vector_of_edge_degrees.append(edge_degree)
            vector_of_stub_degrees.append(stub_degree)
            Q = upcoming_nodes
        Census_of_Nodes.append(vector_of_node_degrees)
        Census_of_Edges.append(vector_of_edge_degrees)
        Census_of_Stubs.append(vector_of_stub_degrees)
    return Census_of_Nodes, Census_of_Edges, Census_of_Stubs


def get_BMatrix_size(list_of_lists):
    max_length = 0
    largest_value = float("-inf")
    for sublist in list_of_lists:
        sublist_length = len(sublist)
        if sublist_length > max_length:
            max_length = sublist_length
        for value in sublist:
            if value > largest_value:
                largest_value = value
    if largest_value < 1:
        return 2, 2
    else:
        return max_length, largest_value + 1


def BMatrix_of_Census(Census_of):
    census_array = deepcopy(Census_of)
    # for signal in census_array:
    #     signal.insert(0, 1)
    matrix_X, matrix_Y = get_BMatrix_size(census_array)
    BMatrix_aggregate = np.zeros((matrix_X, matrix_Y))
    for signal in census_array:
        if len(signal) < matrix_X:
            signal += [0] * (matrix_X - len(signal))
        for row, col in enumerate(signal):
            BMatrix_aggregate[row][col] += 1
    return BMatrix_aggregate


###############################################################################

graph6_string = "F?qvW"

G = nx.from_graph6_bytes(graph6_string.encode())

G_pos = {
    0: (525, -214),
    1: (602, -648),
    2: (155, -503),
    3: (80, -67),
    4: (701, -406),
    5: (254, -261),
    6: (428, -453),
}

###############################################################################

fig, ax = plt.subplots(figsize=(figure_size, figure_size))
for edge in G.edges():
    x1, y1 = G_pos[edge[0]]
    x2, y2 = G_pos[edge[1]]
    ax.plot(
        [x1, x2],
        [y1, y2],
        "k-",
        linewidth=2.5,
        zorder=0,
    )
x_values = [pos[0] for pos in G_pos.values()]
y_values = [pos[1] for pos in G_pos.values()]
for idx, x_value in enumerate(x_values):
    ax.scatter(
        x_value,
        y_values[idx],
        s=1500,
        c=new_tableau_10[idx],
        edgecolors="none",
        zorder=1,
    )
    ax.annotate(
        "N" + str(idx + 1),
        (
            x_value,
            y_values[idx] - 0.015,
        ),
        color="w",
        size=17,
        va="center",
        ha="center",
        zorder=2,
    )
ax.set_position([0, 0, 1, 1])
ax.margins(0.18)
plt.axis("off")

plt.show(block=False)

# plt.savefig(f"fig_data_structure/NL_zp.pdf", format="pdf", dpi=300)
# plt.close()

###############################################################################

Census_of_Nodes, _, _ = BFS_Census(G)

# print()
# for idx, trajectory in enumerate(Census_of_Nodes):
#     print(idx, "\t", trajectory)
# print()

fig, ax = plt.subplots(figsize=(figure_size, figure_size))

for idx, trajectory in enumerate(Census_of_Nodes):
    ax.plot(
        list(range(1, len(Census_of_Nodes[idx]) + 1)),
        trajectory,
        "-",
        color=new_tableau_10[idx],
        markersize=10,
        linewidth=6,
        zorder=-idx,
        solid_capstyle="round",
    )
ax.set_xlabel("Hop Number")
ax.set_ylabel("Node Degree")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()

plt.show(block=False)

# plt.savefig(f"fig_data_structure/CN_zp.pdf", format="pdf", dpi=300)
# plt.close()

###############################################################################

BMatrix_of_Node = BMatrix_of_Census(Census_of_Nodes)
BMatrix_of_Node[BMatrix_of_Node == 0.0] = np.nan

BN = BMatrix_of_Node

fig, ax = plt.subplots(figsize=(figure_size, figure_size))
ax.pcolor(BN.T, cmap="inferno_r", edgecolors="k")
ax.set_xlabel("Hop Number")
ax.set_ylabel("Node Degree")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.set_aspect("equal")
ax.grid(True, which="both", color="k", linewidth=1)

cbar = fig.colorbar(ax.pcolormesh(BN.T, cmap="inferno_r"), pad=0.12)
cbar.set_label("Frequency of Occurrence")
cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# cbar.ax.yaxis.set_label_position("left")
# cbar.ax.yaxis.set_ticks_position("left")
plt.tight_layout()

plt.show(block=False)

# plt.savefig(f"fig_data_structure/BN_zp.pdf", format="pdf", dpi=300)
# plt.close()

###############################################################################

fig, ax = plt.subplots(figsize=(figure_size, figure_size))
x_values = []
y_values = []
z_values = []
z_stores = []
for x, row in enumerate(BMatrix_of_Node):
    for y, value in enumerate(row):
        x_values.append(x + 1)
        y_values.append(y)
        z_values.append(value)
        z_stores.append(value * 0)
ax.scatter(
    x_values,
    y_values,
    s=500,
    c=z_stores,
    marker="s",
    # cmap="inferno_r",
    cmap="Greys",
    edgecolors="k",
    linewidth=2,
    zorder=1,
)

for idx, z_value in enumerate(z_values):
    if not np.isnan(z_value):
        ax.annotate(
            int(z_value),
            (
                x_values[idx],
                y_values[idx] - 0.015,
            ),
            color="k",
            size=10,
            va="center",
            ha="center",
            zorder=2,
        )


for idx, trajectory in enumerate(Census_of_Nodes):
    ax.plot(
        list(range(1, len(Census_of_Nodes[idx]) + 1)),
        trajectory,
        c="k",
        markersize=10,
        linewidth=2,
        zorder=0,
    )
ax.plot(
    [3, 4],
    [0, 0],
    c="k",
    markersize=10,
    linewidth=2,
    zorder=0,
)
ax.set_xlabel("Hop Number")
ax.set_ylabel("Node Degree")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.margins(0.1)
plt.tight_layout()

plt.show(block=False)

# plt.savefig(f"fig_data_structure/CN_aggregate_zp.pdf", format="pdf", dpi=300)
# plt.close()

###############################################################################

fig = plt.figure(figsize=(figure_size, figure_size))
ax = fig.add_subplot(111)
ax.axis([0, 100, 0, 100])

linespacing = 0
for i, line in enumerate(Census_of_Nodes):
    ax.text(
        6,
        94.5 - linespacing,
        "     " + str(line),
        ha="left",
        va="top",
        fontsize=23,
    )
    linespacing += 13.15

linespacing = 0
for i, line in enumerate(Census_of_Nodes):
    ax.text(
        6,
        94.5 - linespacing,
        "N" + str(i + 1),
        ha="left",
        va="top",
        fontsize=23,
        color=new_tableau_10[i],
    )
    linespacing += 13.15
plt.axis("off")

plt.show(block=False)

# plt.savefig(f"fig_data_structure/TXT_zp.pdf", format="pdf", dpi=300)
# plt.close()

###############################################################################

plt.show()