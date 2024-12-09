# by Matias I. Bofarull Oddo - 2024.03.18

import csv
import json
from collections import Counter
from copy import deepcopy
from pprint import pprint
from random import shuffle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
from matplotlib.font_manager import FontProperties

plt.rcParams.update(
    {
        "font.sans-serif": "Consolas",
        "font.weight": "bold",
        "font.size": 14,
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
    Census_of_Stubs = []
    for source_node in D.keys():
        current_nodes = [source_node]
        seen_nodes = set(current_nodes)
        seen_stubs = set()
        count_nodes = []
        count_stubs = []
        while len(current_nodes) > 0:
            hop_node_count = 0
            hop_stub_count = 0
            current_stubs = set()
            next_nodes = []
            for node in current_nodes:
                neighbors = D[node].keys()
                for neighbor in neighbors:
                    if neighbor not in seen_nodes:
                        next_nodes.append(neighbor)
                        hop_node_count += 1
                    stub = (node, neighbor)
                    if stub not in seen_stubs:
                        hop_stub_count += 1
                    seen_nodes.add(neighbor)
                    current_stubs.add(stub)
                    seen_stubs.add(stub)
            for stub in current_stubs:
                seen_stubs.add((stub[1], stub[0]))
            count_nodes.append(hop_node_count)
            count_stubs.append(hop_stub_count)
            current_nodes = next_nodes
        Census_of_Nodes.append(count_nodes)
        Census_of_Stubs.append(count_stubs)
    return Census_of_Nodes, Census_of_Stubs


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

Census_of_Nodes, _ = BFS_Census(G)
BMatrix_of_Node, IDX_of_Node = BMatrix_of_Census(Census_of_Nodes)
BMatrix_of_Node[BMatrix_of_Node == 0.0] = np.nan

###############################################################################

edge_list = []

node_degrees = dict(G.degree())
nodes = np.array(list(node_degrees.keys()))
np.random.shuffle(nodes)

sorted_nodes = sorted(nodes, key=lambda x: node_degrees[x], reverse=False)
node_mapping = {node: i for i, node in enumerate(sorted_nodes)}

for edge in G.edges():
    edge_list.append((node_mapping[edge[0]], node_mapping[edge[1]]))

fig, ax = plt.subplots(figsize=(figure_size, figure_size))

for edge in edge_list:
    ax.scatter(edge[0], edge[1], s=5, color="k")
    ax.scatter(edge[1], edge[0], s=5, color="k")

ax.set_xticks(range(1, len(sorted_nodes), 75))
ax.set_yticks(range(1, len(sorted_nodes), 75))

plt.tight_layout()

# plt.show()

plt.savefig(f"Intro_02.png", format="png", dpi=800)
plt.close()

###############################################################################

node_degrees = []

for row in Census_of_Nodes:
    node_degrees.append(row[0])

freq_degrees = Counter(node_degrees)

x_values = []
y_values = []

for key, value in freq_degrees.items():
    x_values.append(key)
    y_values.append(value)

fig, ax = plt.subplots(figsize=(figure_size, figure_size * 0.66))

for i in range(len(x_values)):
    ax.plot(
        [x_values[i], x_values[i]],
        [0, y_values[i]],
        color="k",
        solid_capstyle="round",
        linewidth=6,
    )

ax.set_ylim(-6, 83)
ax.set_xticks(range(1, max(x_values) + 1, 4))

# ax.set_xlabel("Node Degree")
# ax.set_ylabel("Frequency of Occurrence")

plt.tight_layout()

# plt.show()

plt.savefig(f"Intro_03.png", format="png", dpi=800)
plt.close()

###############################################################################

fig = plt.figure(figsize=(figure_size, figure_size))
ax = fig.add_subplot(111)

ax.pcolor(
    np.log(BMatrix_of_Node),
    cmap="inferno_r",
    edgecolors="none",
)

ax.set_ylabel("Hop Number")
ax.set_xlabel("Node Degree")
ax.set_ylim(1, len(BMatrix_of_Node))

ax.set_yticks(range(2, 20))
ax.set_yticklabels([str(i - 2) if i % 2 != 0 else "" for i in range(2, 20)])

plt.tight_layout()

plt.gca().invert_yaxis()

# plt.show(block=False)

plt.savefig(f"Intro_05.png", format="png", dpi=800)
plt.close()

###############################################################################

fig = plt.figure(figsize=(figure_size, figure_size))
ax = fig.add_subplot(111)

sample_row = BMatrix_of_Node[1:3]

ax.imshow(np.log(sample_row), cmap="inferno_r")

ax.set_yticks([])

ax.set_xlim(-0.5, 35.5)
ax.set_ylim(0.5, 1.5)

ax.set_xticks(range(1, max(x_values) + 1, 4))

plt.savefig(f"Intro_04.png", format="png", dpi=800)
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
ax.set_position([0, 0, 1, 1])
ax.margins(0.1)
plt.axis("off")

# plt.show(block=False)

plt.savefig(f"Intro_01.png", format="png", dpi=800)
plt.close()

###############################################################################
