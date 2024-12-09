import csv
from multiprocessing import Pool

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np

plt.rcParams.update(
    {
        "font.sans-serif": "Consolas",
        "font.weight": "bold",
        "font.size": 17,
    }
)


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


def COLLECT_VECTOR(source_node, D):
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
    return (
        vector_of_node_degrees,
        vector_of_edge_degrees,
        vector_of_stub_degrees,
    )


def BFS_CENSUS(G):
    D = nx.to_dict_of_dicts(G)
    Census_Node = []
    Census_Edge = []
    Census_Stub = []
    with Pool(11) as p:
        keys = D.keys()
        vectors = p.starmap(COLLECT_VECTOR, [(key, D) for key in keys])
        for (
            vector_of_node_degrees,
            vector_of_edge_degrees,
            vector_of_stub_degrees,
        ) in vectors:
            Census_Node.append(vector_of_node_degrees)
            Census_Edge.append(vector_of_edge_degrees)
            Census_Stub.append(vector_of_stub_degrees)
    return Census_Node, Census_Edge, Census_Stub


def sigmoid_sizes(num_E):
    sigmoid = 1 / (1 + np.exp(-0.001 * (num_E - 1)))
    sigmoid_node = ((1 - sigmoid) * 20) + 0.01
    sigmoid_edge = ((1 - sigmoid) * 2.2) + 0.05
    sigmoid_alph = ((1 - sigmoid) * 1.8) + 0.1
    sigmoid_alph = np.clip(sigmoid_alph, 0.1, 1)
    return sigmoid_node, sigmoid_edge, sigmoid_alph


def max_Census_value(Census):
    max_Census = 0
    for row in Census:
        for value in row:
            if value > max_Census:
                max_Census = value
    return max_Census


if __name__ == "__main__":

    figure_size = 6
    edge_width = 0.5

    n = 1500

    ###########################################################################

    G_ER = load_CSV_as_G(f"erdos-reyni/topology.csv")
    G_ER_pos = load_CSV_as_G_pos(f"erdos-reyni/embedding.csv")

    ER_CN, ER_CE, ER_CS = BFS_CENSUS(G_ER)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    G_WS = load_CSV_as_G(f"watts-strogatz/topology.csv")
    G_WS_pos = load_CSV_as_G_pos(f"watts-strogatz/embedding.csv")

    WS_CN, WS_CE, WS_CS = BFS_CENSUS(G_WS)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    G_BA = load_CSV_as_G(f"barabasi-albert/topology.csv")
    G_BA_pos = load_CSV_as_G_pos(f"barabasi-albert/embedding.csv")

    BA_CN, BA_CE, BA_CS = BFS_CENSUS(G_BA)

    ###########################################################################

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111)

    max_length = 0

    for i in range(len(BA_CN)):
        ax.plot(
            list(range(len(BA_CN[i]))),
            BA_CN[i],
            c="#4e79a7",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )
        max_length = max(max_length, len(BA_CN[i]))

    for i in range(len(WS_CN)):
        ax.plot(
            list(range(len(WS_CN[i]))),
            WS_CN[i],
            c="#59a14f",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )
        max_length = max(max_length, len(WS_CN[i]))

    for i in range(len(ER_CN)):
        ax.plot(
            list(range(len(ER_CN[i]))),
            ER_CN[i],
            c="#e15759",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )
        max_length = max(max_length, len(ER_CN[i]))

    for i in range(1, max_length + 1):
        ax.axvline(
            x=i - 1,
            color="skyblue",
            alpha=0.4,
            linewidth=4,
            zorder=-1,
        )

    ax.set_xticks(range(0, max_length))
    ax.set_xticklabels([str(i) for i in range(1, max_length + 1)])

    # ax.set_xlabel("Hop Number")
    # ax.set_ylabel("Node Degree")

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    plt.tight_layout()

    plt.savefig(f"TRIPLE_CN.png", format="png", dpi=800)
    plt.close()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111)

    max_length = 0

    for i in range(len(BA_CE)):
        ax.plot(
            list(range(len(BA_CE[i]))),
            BA_CE[i],
            c="#4e79a7",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )
        max_length = max(max_length, len(BA_CE[i]))

    for i in range(len(WS_CE)):
        ax.plot(
            list(range(len(WS_CE[i]))),
            WS_CE[i],
            c="#59a14f",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )
        max_length = max(max_length, len(WS_CE[i]))

    for i in range(len(ER_CE)):
        ax.plot(
            list(range(len(ER_CE[i]))),
            ER_CE[i],
            c="#e15759",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )
        max_length = max(max_length, len(ER_CE[i]))

    for i in range(1, max_length + 1):
        ax.axvline(
            x=i - 1,
            color="skyblue",
            alpha=0.4,
            linewidth=4,
            zorder=-1,
        )

    ax.set_xticks(range(0, max_length))
    ax.set_xticklabels([str(i) for i in range(1, max_length + 1)])

    # ax.set_xlabel("Hop Number")
    # ax.set_ylabel("edge Degree")

    plt.tight_layout()

    plt.savefig(f"TRIPLE_CE.png", format="png", dpi=800)
    plt.close()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111)

    max_length = 0

    for i in range(len(BA_CS)):
        ax.plot(
            list(range(len(BA_CS[i]))),
            BA_CS[i],
            c="#4e79a7",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )
        max_length = max(max_length, len(BA_CS[i]))

    for i in range(len(WS_CS)):
        ax.plot(
            list(range(len(WS_CS[i]))),
            WS_CS[i],
            c="#59a14f",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )
        max_length = max(max_length, len(WS_CS[i]))

    for i in range(len(ER_CS)):
        ax.plot(
            list(range(len(ER_CS[i]))),
            ER_CS[i],
            c="#e15759",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )
        max_length = max(max_length, len(ER_CS[i]))

    for i in range(1, max_length + 1):
        ax.axvline(
            x=i - 1,
            color="skyblue",
            alpha=0.4,
            linewidth=4,
            zorder=-1,
        )

    ax.set_xticks(range(0, max_length))
    ax.set_xticklabels([str(i) for i in range(1, max_length + 1)])

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    # ax.set_xlabel("Hop Number")
    # ax.set_ylabel("Stub Degree")

    plt.tight_layout()

    plt.savefig(f"TRIPLE_CS.png", format="png", dpi=800)
    plt.close()

    ###########################################################################

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111)
    for i in range(len(ER_CN)):
        ax.plot(
            ER_CN[i],
            ER_CS[i],
            c="k",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines["top"].set_visible(True)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["left"].set_visible(True)
    plt.gca().spines["right"].set_visible(True)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    plt.savefig(f"CS-CS_ER.png", format="png", dpi=800)
    plt.close()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111)
    for i in range(len(WS_CN)):
        ax.plot(
            WS_CN[i],
            WS_CS[i],
            c="k",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines["top"].set_visible(True)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["left"].set_visible(True)
    plt.gca().spines["right"].set_visible(True)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    plt.savefig(f"CN-CS_WS.png", format="png", dpi=800)
    plt.close()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111)
    for i in range(len(BA_CN)):
        ax.plot(
            BA_CN[i],
            BA_CS[i],
            c="k",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines["top"].set_visible(True)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["left"].set_visible(True)
    plt.gca().spines["right"].set_visible(True)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    plt.savefig(f"CN-CS_BA.png", format="png", dpi=800)
    plt.close()

    ###########################################################################

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111)

    for i in range(len(BA_CN)):
        ax.plot(
            BA_CN[i],
            BA_CE[i],
            c="#4e79a7",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )

    for i in range(len(WS_CN)):
        ax.plot(
            WS_CN[i],
            WS_CE[i],
            c="#59a14f",
            linewidth=0.3,
            alpha=0.2,
            solid_capstyle="round",
        )

    for i in range(len(ER_CN)):
        ax.plot(
            ER_CN[i],
            ER_CE[i],
            c="#e15759",
            linewidth=0.2,
            alpha=0.2,
            solid_capstyle="round",
        )

    # ax.set_xlabel("Node Degree")
    # ax.set_ylabel("Edge Degree")

    plt.tight_layout()

    plt.savefig(f"TRIPLE_CN-CE.png", format="png", dpi=800)
    plt.close()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111)

    for i in range(len(BA_CE)):
        ax.plot(
            BA_CE[i],
            BA_CS[i],
            c="#4e79a7",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )

    for i in range(len(WS_CE)):
        ax.plot(
            WS_CE[i],
            WS_CS[i],
            c="#59a14f",
            linewidth=0.3,
            alpha=0.2,
            solid_capstyle="round",
        )

    for i in range(len(ER_CE)):
        ax.plot(
            ER_CE[i],
            ER_CS[i],
            c="#e15759",
            linewidth=0.2,
            alpha=0.2,
            solid_capstyle="round",
        )

    # ax.set_xlabel("Edge Degree")
    # ax.set_ylabel("Stub Degree")

    plt.tight_layout()

    plt.savefig(f"TRIPLE_CE-CS.png", format="png", dpi=800)
    plt.close()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111)

    for i in range(len(BA_CN)):
        ax.plot(
            BA_CN[i],
            BA_CS[i],
            c="#4e79a7",
            linewidth=0.3,
            alpha=0.3,
            solid_capstyle="round",
        )

    for i in range(len(WS_CN)):
        ax.plot(
            WS_CN[i],
            WS_CS[i],
            c="#59a14f",
            linewidth=0.3,
            alpha=0.2,
            solid_capstyle="round",
        )

    for i in range(len(ER_CN)):
        ax.plot(
            ER_CN[i],
            ER_CS[i],
            c="#e15759",
            linewidth=0.2,
            alpha=0.2,
            solid_capstyle="round",
        )

    # ax.set_xlabel("Node Degree")
    # ax.set_ylabel("Stub Degree")

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    plt.tight_layout()

    plt.savefig(f"TRIPLE_CN-CS.png", format="png", dpi=800)
    plt.close()

    ###########################################################################

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111)
    for edge in G_ER.edges():
        x1, y1 = G_ER_pos[edge[0]]
        x2, y2 = G_ER_pos[edge[1]]
        ax.plot(
            [x1, x2],
            [y1, y2],
            c="silver",
            alpha=0.5,
            linewidth=0.3,
            zorder=-10,
        )
    x_values = [pos[0] for pos in G_ER_pos.values()]
    y_values = [pos[1] for pos in G_ER_pos.values()]
    ax.scatter(
        x_values,
        y_values,
        s=100,
        alpha=0.7,
        facecolors="#e15759",
        edgecolors="none",
        zorder=10,
    )
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    plt.savefig(f"NL_ER.png", format="png", dpi=800)
    plt.close()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111)
    for edge in G_WS.edges():
        x1, y1 = G_WS_pos[edge[0]]
        x2, y2 = G_WS_pos[edge[1]]
        ax.plot(
            [x1, x2],
            [y1, y2],
            c="silver",
            alpha=0.5,
            linewidth=0.3,
            zorder=-10,
        )
    x_values = [pos[0] for pos in G_WS_pos.values()]
    y_values = [pos[1] for pos in G_WS_pos.values()]
    ax.scatter(
        x_values,
        y_values,
        s=100,
        alpha=0.7,
        facecolors="#59a14f",
        edgecolors="none",
        zorder=10,
    )
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    plt.savefig(f"NL_WS.png", format="png", dpi=800)
    plt.close()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111)
    for edge in G_BA.edges():
        x1, y1 = G_BA_pos[edge[0]]
        x2, y2 = G_BA_pos[edge[1]]
        ax.plot(
            [x1, x2],
            [y1, y2],
            c="silver",
            alpha=0.5,
            linewidth=0.3,
            zorder=-10,
        )
    x_values = [pos[0] for pos in G_BA_pos.values()]
    y_values = [pos[1] for pos in G_BA_pos.values()]
    ax.scatter(
        x_values,
        y_values,
        s=100,
        alpha=0.7,
        facecolors="#4e79a7",
        edgecolors="none",
        zorder=10,
    )
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    plt.savefig(f"NL_BA.png", format="png", dpi=800)
    plt.close()

###############################################################################
