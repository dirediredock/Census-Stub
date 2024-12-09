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
colormap = "viridis_r"


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

    filenames = [str(i + 1).zfill(2) for i in range(81)]

    edge_width = 1
    edge_alpha = 0.05

    S = 0.05
    B = 0.95

    for filename in filenames:

        G = nx.barabasi_albert_graph(500, 2)
        G_pos = nx.spring_layout(G, k=0.0005, iterations=100)

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
                linewidth=0.4,
                alpha=0.4,
                zorder=0,
            )
        x_values = [pos[0] for pos in G_pos.values()]
        y_values = [pos[1] for pos in G_pos.values()]
        for idx, x_value in enumerate(x_values):
            ax.scatter(
                x_value,
                y_values[idx],
                s=3,
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
            f"FIG_LocalGlobal/NL_{filename}.png",
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
            f"FIG_LocalGlobal/CN_{filename}.png",
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
            f"FIG_LocalGlobal/CE_{filename}.png",
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
            f"FIG_LocalGlobal/CS_{filename}.png",
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
            f"FIG_LocalGlobal/BN_{filename}.png",
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
            f"FIG_LocalGlobal/BE_{filename}.png",
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
            f"FIG_LocalGlobal/BS_{filename}.png",
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
            f"FIG_LocalGlobal/CN_CE_{filename}.png",
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
            f"FIG_LocalGlobal/CE_CS_{filename}.png",
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
            f"FIG_LocalGlobal/CN_CS_{filename}.png",
            format="png",
            dpi=400,
        )
        plt.close()

        print(f"Finished {filename}\n")

###############################################################################
