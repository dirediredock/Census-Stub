# by Matias I. Bofarull Oddo - 2024.03.18

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
import networkx as nx
import numpy as np
from copy import deepcopy
from collections import Counter
from matplotlib.font_manager import FontProperties
import json
from pprint import pprint

plt.rcParams.update(
    {
        "font.sans-serif": "Consolas",
        "font.weight": "bold",
        "font.size": 17,
    }
)

figure_size = 6
colormap = "inferno_r"


def load_CSV_as_G(csv_filename):
    G = nx.Graph()
    with open(csv_filename, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            G.add_edge(int(row[0]), int(row[1]))
    return G


def load_CSV_as_G_pos(csv_filename):
    G_pos = {}
    with open(csv_filename, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        for idx, coords_XY in enumerate(csv_reader):
            G_pos[idx] = (float(coords_XY[0]), float(coords_XY[1]))
    return G_pos


def sigmoid_sizes(num_E):
    sigmoid = 1 / (1 + np.exp(-0.001 * (num_E - 1)))
    sigmoid_node = ((1 - sigmoid) * 20) + 0.01
    sigmoid_edge = ((1 - sigmoid) * 2.2) + 0.05
    sigmoid_alph = ((1 - sigmoid) * 1.8) + 0.1
    sigmoid_alph = np.clip(sigmoid_alph, 0.1, 1)
    return sigmoid_node, sigmoid_edge, sigmoid_alph


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
    for signal in census_array:
        signal.insert(0, 1)
        signal.insert(0, 0)
    matrix_X, matrix_Y = get_BMatrix_size(census_array)
    BMatrix_aggregate = np.zeros((matrix_X, matrix_Y))
    BMatrix_idx = [[[] for _ in range(matrix_Y)] for _ in range(matrix_X)]
    for idx, signal in enumerate(census_array):
        if len(signal) < matrix_X:
            signal += [0] * (matrix_X - len(signal))
        for row, col in enumerate(signal):
            BMatrix_aggregate[row][col] += 1
            BMatrix_idx[row][col].append(idx)
    return BMatrix_aggregate, BMatrix_idx


###############################################################################

G = load_CSV_as_G("topology.csv")
G_pos = load_CSV_as_G_pos("embedding.csv")

node_sizes, edge_width, edge_alpha = sigmoid_sizes(G.number_of_edges())

###############################################################################

Census_of_Nodes, _, Census_of_Stubs = BFS_Census(G)

BMatrix_of_Node, IDX_of_Node = BMatrix_of_Census(Census_of_Nodes)
BMatrix_of_Stub, IDX_of_Stub = BMatrix_of_Census(Census_of_Stubs)

BMatrix_of_Node[BMatrix_of_Node == 0.0] = np.nan
BMatrix_of_Stub[BMatrix_of_Stub == 0.0] = np.nan

###############################################################################

selected_cells = [
    [16, 20],
    [15, 20],
    [14, 20],
]

###############################################################################

highlight_idx = []

for cell in selected_cells:
    highlight_idx.append(IDX_of_Node[cell[0]][cell[1]])

highlight_idx = set([item for sublist in highlight_idx for item in sublist])

print(highlight_idx)

###############################################################################

fig = plt.figure(figsize=(figure_size, figure_size))
ax = fig.add_subplot(111)
ax.pcolor(
    np.log(BMatrix_of_Node.T),
    cmap="inferno_r",
    edgecolors="none",
)

# for cell in selected_cells:
#     cell_row = cell[0]
#     cell_col = cell[1]
#     ax.add_patch(
#         plt.Rectangle(
#             (cell_row, cell_col),
#             1,
#             1,
#             fill=False,
#             edgecolor="#4e79a7",
#             linewidth=5,
#         )
#     )

# # for i, row in enumerate(BMatrix_of_Node):
# #     for j, cell in enumerate(row):
# #         if not np.isnan(cell):
# #             ax.add_patch(
# #                 plt.Rectangle(
# #                     (i, j),
# #                     1,
# #                     1,
# #                     fill=False,
# #                     edgecolor="whitesmoke",
# #                     linewidth=3,
# #                     zorder=-10,
# #                 )
# #             )

ax.set_xlabel("Hop Number")
ax.set_ylabel("Node Degree")
ax.set_xlim(1, len(BMatrix_of_Node))

ax.set_xticks(range(2, 20))
ax.set_xticklabels([str(i - 2) if i % 2 != 0 else "" for i in range(2, 20)])

plt.tight_layout()

# plt.show(block=False)

plt.savefig(f"Echo_01.png", format="png", dpi=800)
plt.close()

###############################################################################

fig = plt.figure(figsize=(figure_size, figure_size))
ax = fig.add_subplot(111)

max_length = 0

for i in range(len(Census_of_Nodes)):
    if i in highlight_idx:
        ax.plot(
            list(range(1, len(Census_of_Nodes[i]) + 1)),
            Census_of_Nodes[i],
            c="#4e79a7",
            solid_capstyle="round",
            linewidth=edge_width + 3,
            zorder=10,
        )
    else:
        ax.plot(
            list(range(1, len(Census_of_Nodes[i]) + 1)),
            Census_of_Nodes[i],
            c="#bab0ac",
            linewidth=edge_width,
            # alpha=edge_alpha,
            zorder=0,
        )
    if len(Census_of_Nodes[i]) > max_length:
        max_length = len(Census_of_Nodes[i])

    for i in range(1, max_length + 1):
        ax.axvline(
            x=i,
            color="#d4f1f4",
            linewidth=2,
            zorder=-100,
        )

ax.set_xlabel("Hop Number")
ax.set_ylabel("Node Degree")

ax.set_xticks(range(1, max_length + 1))
ax.set_xticklabels([str(i) if i % 2 != 0 else "" for i in range(1, max_length + 1)])

plt.tight_layout()

# plt.show(block=False)

plt.savefig(f"Echo_02.png", format="png", dpi=800)
plt.close()

###############################################################################

fig = plt.figure(figsize=(figure_size, figure_size))
ax = fig.add_subplot(111)

max_length = 0

for i in range(len(Census_of_Stubs)):
    if i in highlight_idx:
        ax.plot(
            list(range(1, len(Census_of_Stubs[i]) + 1)),
            Census_of_Stubs[i],
            c="#4e79a7",
            solid_capstyle="round",
            linewidth=edge_width + 3,
            zorder=10,
        )
    else:
        ax.plot(
            list(range(1, len(Census_of_Stubs[i]) + 1)),
            Census_of_Stubs[i],
            c="#bab0ac",
            linewidth=edge_width,
            # alpha=edge_alpha,
            zorder=0,
        )
    if len(Census_of_Nodes[i]) > max_length:
        max_length = len(Census_of_Nodes[i])

    for i in range(1, max_length + 1):
        ax.axvline(
            x=i,
            color="#d4f1f4",
            linewidth=2,
            zorder=-100,
        )

ax.set_xlabel("Hop Number")
ax.set_ylabel("Stub Degree")

ax.set_xticks(range(1, max_length + 1))
ax.set_xticklabels([str(i) if i % 2 != 0 else "" for i in range(1, max_length + 1)])

plt.tight_layout()

# plt.show(block=False)

plt.savefig(f"Echo_03.png", format="png", dpi=800)
plt.close()

###############################################################################

fig, ax = plt.subplots(figsize=(figure_size, figure_size))
if nx.is_planar(G):
    NL_alpha = 0.5
else:
    NL_alpha = edge_alpha
for edge in G.edges():
    x1, y1 = G_pos[edge[0]]
    x2, y2 = G_pos[edge[1]]
    ax.plot(
        [x1, x2],
        [y1, y2],
        "k-",
        alpha=NL_alpha,
        linewidth=edge_width,
    )
x_values = [pos[0] for pos in G_pos.values()]
y_values = [pos[1] for pos in G_pos.values()]

for idx, node in enumerate(x_values):
    ax.scatter(
        x_values[idx],
        y_values[idx],
        s=node_sizes,
        facecolors="k",
        edgecolors="none",
    )
    if idx in highlight_idx:
        ax.scatter(
            x_values[idx],
            y_values[idx],
            s=node_sizes * 40,
            facecolors="none",
            edgecolors="#4e79a7",
            linewidth=5,
            zorder=-10,
        )
ax.set_position([0, 0, 1, 1])
ax.margins(0.1)
plt.axis("off")

# plt.show(block=False)

plt.savefig(f"Echo_05.png", format="png", dpi=800)
plt.close()

###############################################################################

fig = plt.figure(figsize=(figure_size, figure_size))
ax = fig.add_subplot(111)
for i in range(len(Census_of_Nodes)):
    if i in highlight_idx:
        ax.plot(
            Census_of_Nodes[i],
            Census_of_Stubs[i],
            c="#4e79a7",
            solid_capstyle="round",
            linewidth=edge_width + 3,
            zorder=10,
        )
    else:
        ax.plot(
            Census_of_Nodes[i],
            Census_of_Stubs[i],
            c="#bab0ac",
            linewidth=edge_width,
            alpha=edge_alpha,
            zorder=0,
        )
ax.set_xlabel("Node Degree")
ax.set_ylabel("Stub Degree")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()

# plt.show(block=False)

plt.savefig(f"Echo_04.png", format="png", dpi=800)
plt.close()

###############################################################################

# plt.show()
