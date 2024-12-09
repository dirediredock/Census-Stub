# by Matias I. Bofarull Oddo - 2023.09.19

from random import sample as shuffle
from math import comb
import networkx as nx
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


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


if __name__ == "__main__":

    figure_size = 6

    for EDGE in range(12):

        print(EDGE + 1)

        G = nx.Graph()

        number_of_nodes = 13

        for i in range(number_of_nodes - 1):
            G.add_edge(i, i + 1)

        G.add_edge(0, number_of_nodes - 1)

        G.add_edge(0, number_of_nodes - (EDGE + 1))

        G_pos = nx.circular_layout(G)

        if EDGE < number_of_nodes - 2:
            G.remove_edge(0, 1)

        print("\tdiameter", nx.diameter(G))
        print("\tradius", nx.radius(G))

        color_palette = plt.cm.viridis(np.linspace(0, 1, number_of_nodes))

        Loop_CN, Loop_CE, Loop_CS = BFS_CENSUS(G)

        #######################################################################

        fig, ax = plt.subplots(figsize=(figure_size, figure_size))
        for edge in G.edges():
            x1, y1 = G_pos[edge[0]]
            x2, y2 = G_pos[edge[1]]
            ax.plot(
                [x1, x2],
                [y1, y2],
                "k-",
                linewidth=5,
                zorder=0,
            )
        x_values = [pos[0] for pos in G_pos.values()]
        y_values = [pos[1] for pos in G_pos.values()]
        for idx, x_value in enumerate(x_values):
            ax.scatter(
                x_value,
                y_values[idx],
                s=700,
                color=color_palette[idx],
                zorder=1,
            )

        plt.xticks([])
        plt.yticks([])
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        plt.savefig(
            f"output/Loop_NL{str(EDGE+1).zfill(2)}.png",
            format="png",
            dpi=800,
        )
        plt.close()

        #######################################################################

        fig, ax = plt.subplots(figsize=(figure_size, figure_size))

        max_length = 0

        for idx, trajectory in enumerate(Loop_CN):
            ax.plot(
                list(range(1, len(Loop_CN[idx]) + 1)),
                Loop_CS[idx],
                "-",
                color=color_palette[idx],
                markersize=10,
                linewidth=8,
                zorder=-idx,
                solid_capstyle="round",
            )
            max_length = max(max_length, len(Loop_CN[idx]))

        for i in range(1, max_length + 1):
            ax.axvline(
                x=i,
                color="skyblue",
                alpha=0.4,
                linewidth=5,
                zorder=-100,
            )

        ax.set_ylim([-0.5, 3.5])

        plt.xticks([])
        plt.yticks([])
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        plt.savefig(
            f"output/Loop_PS{str(EDGE+1).zfill(2)}.png",
            format="png",
            dpi=800,
        )
        plt.close()

###############################################################################
