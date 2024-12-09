import csv
from collections import Counter
from copy import deepcopy
from multiprocessing import Pool

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


plt.rcParams.update(
    {
        "font.sans-serif": "Consolas",
        "font.weight": "bold",
        "font.size": 12,
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


def COLLECT_VECTOR(source_node, D):
    Q = [source_node]
    visited_Node = set(Q)
    visited_Edge = set()
    visited_Stub = set()
    vector_of_node_degrees = []
    vector_of_edge_degrees = []
    vector_of_stub_degrees = []
    while len(Q) > 0:
        node_degree = 0
        edge_degree = 0
        stub_degree = 0
        current_Stub = set()
        upcoming_Node = []
        for node in Q:
            neighbors = D[node].keys()
            for neighbor in neighbors:
                if neighbor not in visited_Node:
                    upcoming_Node.append(neighbor)
                    node_degree += 1
                visited_Node.add(neighbor)
                edge = (min(node, neighbor), max(node, neighbor))
                if edge not in visited_Edge:
                    edge_degree += 1
                visited_Edge.add(edge)
                stub = (node, neighbor)
                if stub not in visited_Stub:
                    stub_degree += 1
                current_Stub.add(stub)
                visited_Stub.add(stub)
        for stub in current_Stub:
            visited_Stub.add((stub[1], stub[0]))
        vector_of_node_degrees.append(node_degree)
        vector_of_edge_degrees.append(edge_degree)
        vector_of_stub_degrees.append(stub_degree)
        Q = upcoming_Node
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


###############################################################################

if __name__ == "__main__":

    filenames = [
        "erdos-renyi",
        "watts-strogatz",
        "grid-14-by-14",
        "spiderweb-3-100",
        "tree-2-7",
        "star-14",
        "fox",
        "power-grid",
        "bacteria-metabolism",
        "ER-A",
        "WS-A",
        "grid-56-by-56",
        "spiderweb-5-200",
        "tree-3-6",
        "star-28",
        "camel",
        "bible",
        "human-metabolism",
        "ER-C",
        "WS-C",
        "equilateral-14",
        "spiderweb-7-300",
        "tree-4-5",
        "star-56",
        "stanford-bunny",
        "les-miserables",
        "protein",
        "barabasi-albert",
        "WS-D",
        "equilateral-56",
        "example-graph",
        "tree-5-4",
        "percolation-A",
        "fibonacci-sunflower",
        "languages",
        "roundworm",
        "BA-A",
        "random-A",
        "sudoku-2",
        "dodecahedron",
        "tree-A",
        "percolation-B",
        "karate-club",
        "ten-friends",
        "yeast",
        "BA-B",
        "random-B",
        "sudoku-3",
        "desargues",
        "tree-B",
        "percolation-C",
        "college-football",
        "crime",
        "ecosystem-A",
        "BA-C",
        "LFR-A",
        "sudoku-4",
        "frucht",
        "tree-C",
        "percolation-mesh-A",
        "network-science",
        "blogs",
        "ecosystem-B",
        "stochastic-block-model",
        "LFR-B",
        "clique-15",
        "petersen",
        "tree-7-binomial",
        "percolation-mesh-B",
        "collaboration",
        "email",
        "ecosystem-C",
        "SBM",
        "DGM",
        "clique-43",
        "tree-1-8",
        "tree-14-binomial",
        "percolation-mesh-C",
        "euroroad",
        "infectious-expo",
        "ecosystem-D",
    ]

    S = 0.05
    B = 0.95

    for filename in filenames:

        print(f"Processing {filename}")

        G = load_CSV_as_G(f"DATA/{filename}/topology.csv")
        G_pos = load_CSV_as_G_pos(f"DATA/{filename}/embedding.csv")

        node_sizes, _, edge_alpha = sigmoid_sizes(G.number_of_edges())

        edge_width = 0.5

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        Census_Node, Census_Edge, Census_Stub = BFS_CENSUS(G)

        #######################################################################

        fig, ax = plt.subplots(figsize=(figure_size, figure_size))
        for edge in G.edges():
            x1, y1 = G_pos[edge[0]]
            x2, y2 = G_pos[edge[1]]
            ax.plot(
                [x1, x2],
                [y1, y2],
                "k-",
                linewidth=edge_width,
                alpha=edge_alpha,
                zorder=0,
            )
        x_values = [pos[0] for pos in G_pos.values()]
        y_values = [pos[1] for pos in G_pos.values()]
        for idx, x_value in enumerate(x_values):
            ax.scatter(
                x_value,
                y_values[idx],
                s=node_sizes,
                c="k",
                edgecolors="none",
                zorder=1,
            )

        plt.xticks([])
        plt.yticks([])
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.subplots_adjust(left=S, right=B, top=B, bottom=S)

        ax.set_xlabel(
            filename,
            fontsize=15,
            fontweight="bold",
            fontname="Arial",
        )

        plt.savefig(
            f"FIG_VisualEncodingPipeline/{filename}_NL.png",
            format="png",
            dpi=400,
        )
        plt.close()

        #######################################################################

        fig, ax = plt.subplots(figsize=(figure_size, figure_size))

        max_length = 0

        for idx, trajectory in enumerate(Census_Node):
            ax.plot(
                list(range(1, len(Census_Node[idx]) + 1)),
                trajectory,
                "-",
                color="k",
                alpha=edge_alpha,
                linewidth=edge_width,
                solid_capstyle="round",
                zorder=1,
            )
            max_length = max(max_length, len(Census_Node[idx]))

        for i in range(1, max_length + 1):
            ax.axvline(
                x=i,
                color="skyblue",
                alpha=0.4,
                linewidth=2,
                zorder=-1,
            )

        plt.xticks([])
        plt.yticks([])
        plt.gca().spines["top"].set_visible(True)
        plt.gca().spines["bottom"].set_visible(True)
        plt.gca().spines["left"].set_visible(True)
        plt.gca().spines["right"].set_visible(True)
        plt.subplots_adjust(left=S, right=B, top=B, bottom=S)

        ax.set_xlabel(
            filename,
            fontsize=15,
            fontweight="bold",
            fontname="Arial",
        )

        plt.savefig(
            f"FIG_VisualEncodingPipeline/{filename}_CN.png",
            format="png",
            dpi=400,
        )
        plt.close()

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        fig, ax = plt.subplots(figsize=(figure_size, figure_size))

        max_length = 0

        for idx, trajectory in enumerate(Census_Edge):
            ax.plot(
                list(range(1, len(Census_Edge[idx]) + 1)),
                trajectory,
                "-",
                color="k",
                alpha=edge_alpha,
                linewidth=edge_width,
                solid_capstyle="round",
                zorder=1,
            )
            max_length = max(max_length, len(Census_Node[idx]))

        for i in range(1, max_length + 1):
            ax.axvline(
                x=i,
                color="skyblue",
                alpha=0.4,
                linewidth=2,
                zorder=-1,
            )

        plt.xticks([])
        plt.yticks([])
        plt.gca().spines["top"].set_visible(True)
        plt.gca().spines["bottom"].set_visible(True)
        plt.gca().spines["left"].set_visible(True)
        plt.gca().spines["right"].set_visible(True)
        plt.subplots_adjust(left=S, right=B, top=B, bottom=S)

        ax.set_xlabel(
            filename,
            fontsize=15,
            fontweight="bold",
            fontname="Arial",
        )

        plt.savefig(
            f"FIG_VisualEncodingPipeline/{filename}_CE.png",
            format="png",
            dpi=400,
        )
        plt.close()

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        fig, ax = plt.subplots(figsize=(figure_size, figure_size))

        max_length = 0

        for idx, trajectory in enumerate(Census_Stub):
            ax.plot(
                list(range(1, len(Census_Stub[idx]) + 1)),
                trajectory,
                "-",
                color="k",
                alpha=edge_alpha,
                linewidth=edge_width,
                solid_capstyle="round",
                zorder=1,
            )
            max_length = max(max_length, len(Census_Node[idx]))

        for i in range(1, max_length + 1):
            ax.axvline(
                x=i,
                color="skyblue",
                alpha=0.4,
                linewidth=2,
                zorder=-1,
            )

        plt.xticks([])
        plt.yticks([])
        plt.gca().spines["top"].set_visible(True)
        plt.gca().spines["bottom"].set_visible(True)
        plt.gca().spines["left"].set_visible(True)
        plt.gca().spines["right"].set_visible(True)
        plt.subplots_adjust(left=S, right=B, top=B, bottom=S)

        ax.set_xlabel(
            filename,
            fontsize=15,
            fontweight="bold",
            fontname="Arial",
        )

        plt.savefig(
            f"FIG_VisualEncodingPipeline/{filename}_CS.png",
            format="png",
            dpi=400,
        )
        plt.close()

        #######################################################################

        BMatrix = BMatrix_of_Census(Census_Node)
        BMatrix[BMatrix == 0.0] = np.nan

        fig, ax = plt.subplots(figsize=(figure_size, figure_size))
        ax.pcolor(np.log10(BMatrix.T), cmap=colormap, edgecolors="none")

        for i, row in enumerate(BMatrix):
            for j, cell in enumerate(row):
                if not np.isnan(cell):
                    ax.add_patch(
                        plt.Rectangle(
                            (i, j),
                            1,
                            1,
                            fill=False,
                            edgecolor="whitesmoke",
                            linewidth=3,
                            zorder=-10,
                        )
                    )

        plt.xticks([])
        plt.yticks([])
        plt.gca().spines["top"].set_visible(True)
        plt.gca().spines["bottom"].set_visible(True)
        plt.gca().spines["left"].set_visible(True)
        plt.gca().spines["right"].set_visible(True)
        plt.subplots_adjust(left=S, right=B, top=B, bottom=S)

        ax.set_xlabel(
            filename,
            fontsize=15,
            fontweight="bold",
            fontname="Arial",
        )

        plt.savefig(
            f"FIG_VisualEncodingPipeline/{filename}_BN.png",
            format="png",
            dpi=400,
        )
        plt.close()

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        BMatrix = BMatrix_of_Census(Census_Edge)
        BMatrix[BMatrix == 0.0] = np.nan

        fig, ax = plt.subplots(figsize=(figure_size, figure_size))
        ax.pcolor(np.log10(BMatrix.T), cmap=colormap, edgecolors="none")

        for i, row in enumerate(BMatrix):
            for j, cell in enumerate(row):
                if not np.isnan(cell):
                    ax.add_patch(
                        plt.Rectangle(
                            (i, j),
                            1,
                            1,
                            fill=False,
                            edgecolor="whitesmoke",
                            linewidth=3,
                            zorder=-10,
                        )
                    )

        plt.xticks([])
        plt.yticks([])
        plt.gca().spines["top"].set_visible(True)
        plt.gca().spines["bottom"].set_visible(True)
        plt.gca().spines["left"].set_visible(True)
        plt.gca().spines["right"].set_visible(True)
        plt.subplots_adjust(left=S, right=B, top=B, bottom=S)

        ax.set_xlabel(
            filename,
            fontsize=15,
            fontweight="bold",
            fontname="Arial",
        )

        plt.savefig(
            f"FIG_VisualEncodingPipeline/{filename}_BE.png",
            format="png",
            dpi=400,
        )
        plt.close()

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        BMatrix = BMatrix_of_Census(Census_Stub)
        BMatrix[BMatrix == 0.0] = np.nan

        fig, ax = plt.subplots(figsize=(figure_size, figure_size))
        ax.pcolor(np.log10(BMatrix.T), cmap=colormap, edgecolors="none")

        for i, row in enumerate(BMatrix):
            for j, cell in enumerate(row):
                if not np.isnan(cell):
                    ax.add_patch(
                        plt.Rectangle(
                            (i, j),
                            1,
                            1,
                            fill=False,
                            edgecolor="whitesmoke",
                            linewidth=3,
                            zorder=-10,
                        )
                    )

        plt.xticks([])
        plt.yticks([])
        plt.gca().spines["top"].set_visible(True)
        plt.gca().spines["bottom"].set_visible(True)
        plt.gca().spines["left"].set_visible(True)
        plt.gca().spines["right"].set_visible(True)
        plt.subplots_adjust(left=S, right=B, top=B, bottom=S)

        ax.set_xlabel(
            filename,
            fontsize=15,
            fontweight="bold",
            fontname="Arial",
        )

        plt.savefig(
            f"FIG_VisualEncodingPipeline/{filename}_BS.png",
            format="png",
            dpi=400,
        )
        plt.close()

        #######################################################################

        fig, ax = plt.subplots(figsize=(figure_size, figure_size))

        for idx, trajectory in enumerate(Census_Node):
            ax.plot(
                trajectory,
                Census_Edge[idx],
                "-",
                color="k",
                alpha=edge_alpha,
                linewidth=edge_width,
                zorder=-idx,
                solid_capstyle="round",
            )

        plt.xticks([])
        plt.yticks([])
        plt.gca().spines["top"].set_visible(True)
        plt.gca().spines["bottom"].set_visible(True)
        plt.gca().spines["left"].set_visible(True)
        plt.gca().spines["right"].set_visible(True)
        plt.subplots_adjust(left=S, right=B, top=B, bottom=S)

        ax.set_xlabel(
            filename,
            fontsize=15,
            fontweight="bold",
            fontname="Arial",
        )

        plt.savefig(
            f"FIG_VisualEncodingPipeline/{filename}_CN_CE.png",
            format="png",
            dpi=400,
        )
        plt.close()

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        fig, ax = plt.subplots(figsize=(figure_size, figure_size))

        for idx, trajectory in enumerate(Census_Edge):
            ax.plot(
                trajectory,
                Census_Stub[idx],
                "-",
                color="k",
                alpha=edge_alpha,
                linewidth=edge_width,
                zorder=-idx,
                solid_capstyle="round",
            )

        plt.xticks([])
        plt.yticks([])
        plt.gca().spines["top"].set_visible(True)
        plt.gca().spines["bottom"].set_visible(True)
        plt.gca().spines["left"].set_visible(True)
        plt.gca().spines["right"].set_visible(True)
        plt.subplots_adjust(left=S, right=B, top=B, bottom=S)

        ax.set_xlabel(
            filename,
            fontsize=15,
            fontweight="bold",
            fontname="Arial",
        )

        plt.savefig(
            f"FIG_VisualEncodingPipeline/{filename}_CE_CS.png",
            format="png",
            dpi=400,
        )
        plt.close()

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        fig, ax = plt.subplots(figsize=(figure_size, figure_size))

        for idx, trajectory in enumerate(Census_Stub):
            ax.plot(
                Census_Node[idx],
                trajectory,
                "-",
                color="k",
                alpha=edge_alpha,
                linewidth=edge_width,
                zorder=-idx,
                solid_capstyle="round",
            )

        plt.xticks([])
        plt.yticks([])
        plt.gca().spines["top"].set_visible(True)
        plt.gca().spines["bottom"].set_visible(True)
        plt.gca().spines["left"].set_visible(True)
        plt.gca().spines["right"].set_visible(True)
        plt.subplots_adjust(left=S, right=B, top=B, bottom=S)

        ax.set_xlabel(
            filename,
            fontsize=15,
            fontweight="bold",
            fontname="Arial",
        )

        plt.savefig(
            f"FIG_VisualEncodingPipeline/{filename}_CN_CS.png",
            format="png",
            dpi=400,
        )
        plt.close()

        print(f"Finished {filename}\n")

###############################################################################
