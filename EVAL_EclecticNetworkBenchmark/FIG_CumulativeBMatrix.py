import csv
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties

F = FontProperties()
F.set_weight("bold")
F.set_family("Arial")
F.set_size(17)

plt.rcParams.update(
    {
        "font.sans-serif": "Consolas",
        "font.weight": "bold",
        "font.size": 11,
    }
)

node_link = {
    "node_color": "k",
    "edge_color": "k",
    "width": 1,
    "node_size": 10,
    "with_labels": False,
}

###############################################################################


def load_CSV_as_G(csv_filename):
    G = nx.Graph()
    with open(csv_filename, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            G.add_edge(int(row[0]), int(row[1]))
    return G


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


def BFS_Census(G):
    D = nx.to_dict_of_dicts(G)
    Census_of_Nodes = []
    Census_of_Stubs = []
    for source_node in D.keys():
        current_nodes = [source_node]
        seen_nodes = set(current_nodes)
        seen_stubs = set()
        count_nodes = [0]
        count_stubs = [0]
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


def Census_of_Cumulative(list_of_lists):
    Cumulative_Census = []
    for signal in list_of_lists:
        aggregation = 0
        cumulative_signal = []
        for value in signal:
            aggregation += value
            cumulative_signal.append(aggregation)
        Cumulative_Census.append(cumulative_signal[:-1])
    return Cumulative_Census


def BMatrix_of_Census(Census_of):
    census_array = deepcopy(Census_of)
    matrix_X, matrix_Y = get_BMatrix_size(census_array)
    BMatrix_aggregate = np.zeros((matrix_X, matrix_Y))
    for signal in census_array:
        signal[0] = 1
        if len(signal) < matrix_X:
            signal += [0] * (matrix_X - len(signal))
        for row, col in enumerate(signal):
            BMatrix_aggregate[row][col] += 1
    return BMatrix_aggregate


def BMatrix_of_Cumulative(Census_of):
    census_array = deepcopy(Census_of)
    matrix_X, matrix_Y = get_BMatrix_size(census_array)
    BMatrix_aggregate = np.zeros((matrix_X, matrix_Y))
    for signal in census_array:
        signal[0] = 1
        if len(signal) < matrix_X:
            signal += [matrix_Y - 1] * (matrix_X - len(signal))
        for row, col in enumerate(signal):
            BMatrix_aggregate[row][col] += 1
    return BMatrix_aggregate


###############################################################################

G = load_CSV_as_G(f"DATA/network-science/topology.csv")

Census_of_Nodes, Census_of_Stubs = BFS_Census(G)

BMatrix_of_Nodes = BMatrix_of_Census(Census_of_Nodes)
BMatrix_of_Stubs = BMatrix_of_Census(Census_of_Stubs)

Census_of_Cumulative_Nodes = Census_of_Cumulative(Census_of_Nodes)
BMatrix_of_Cumulative_Nodes = BMatrix_of_Cumulative(Census_of_Cumulative_Nodes)

###############################################################################

BMatrix_of_Nodes[BMatrix_of_Nodes == 0.0] = np.nan

temp = np.delete(np.log10(BMatrix_of_Nodes), 0, 0)

fig = plt.figure(figsize=(6, 5.2))
ax = fig.add_subplot(111)
ax.pcolor(temp, cmap="inferno_r")

for i, row in enumerate(temp):
    for j, cell in enumerate(row):
        if not np.isnan(cell):
            ax.add_patch(
                plt.Rectangle(
                    (j, i),
                    1,
                    1,
                    fill=False,
                    edgecolor="whitesmoke",
                    linewidth=3,
                    zorder=-10,
                )
            )

ax.set_xlabel("Node Degree")
ax.set_ylabel("Hop Number")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.invert_yaxis()
plt.tight_layout()

# plt.savefig(f"subset/Fig02.pdf", format="pdf", dpi=300)
# plt.close()

plt.show(block=False)

###############################################################################

GraphPrism = deepcopy(BMatrix_of_Cumulative_Nodes)

temp = np.delete(BMatrix_of_Cumulative_Nodes, 0, 0)

fig = plt.figure(figsize=(6, 5.2))
ax = fig.add_subplot(211)
temp[temp == 0.0] = np.nan
ax.pcolor(np.log10(temp), cmap="inferno_r")

for i, row in enumerate(temp):
    for j, cell in enumerate(row):
        if not np.isnan(cell):
            ax.add_patch(
                plt.Rectangle(
                    (j, i),
                    1,
                    1,
                    fill=False,
                    edgecolor="whitesmoke",
                    linewidth=3,
                    zorder=-10,
                )
            )

ax.set_xlabel("Node Degree (cumulative)")
ax.set_ylabel("Hop Number")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.invert_yaxis()
plt.tight_layout()

# plt.savefig(f"subset/Fig03a.pdf", format="pdf", dpi=300)
# plt.close()

plt.show(block=False)

###############################################################################

GraphPrism = np.delete(GraphPrism, 0, 0)

temp = []

for row in GraphPrism:
    new_row = []
    for i in range(0, len(row), 7):
        sum_of_eight = sum(row[i : i + 7])
        new_row.append(sum_of_eight)
    temp.append(new_row)

data = np.array(temp)

normalized_data = data / np.max(data)

# fig = plt.figure(figsize=(6, 2.6))

ax = fig.add_subplot(212)
temp[temp == 0.0] = np.nan
vmin = np.min(normalized_data[normalized_data > 0])
ax.pcolor(
    normalized_data,
    cmap="inferno_r",
    norm=LogNorm(vmin=vmin, vmax=1),
)

for i, row in enumerate(normalized_data):
    for j, cell in enumerate(row):
        if not cell == 0:
            ax.add_patch(
                plt.Rectangle(
                    (j, i),
                    1,
                    1,
                    fill=False,
                    edgecolor="whitesmoke",
                    linewidth=3,
                    zorder=-10,
                )
            )

ax.set_xlabel("Node Degree (cumulative and binned)")
ax.set_ylabel("Hop Number")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.invert_yaxis()
plt.tight_layout()

fig.text(0.005, 0.95, "A", color="k", fontproperties=F)
fig.text(0.005, 0.465, "B", color="k", fontproperties=F)

plt.show(block=False)

# plt.savefig(f"subset/Fig03.pdf", format="pdf", dpi=300)
# plt.close()

###############################################################################

plt.show()
