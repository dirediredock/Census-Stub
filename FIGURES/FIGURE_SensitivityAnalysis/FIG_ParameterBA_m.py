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


def sigmoid_sizes(num_E):
    sigmoid = 1 / (1 + np.exp(-0.001 * (num_E - 1)))
    sigmoid_node = ((1 - sigmoid) * 20) + 0.01
    sigmoid_edge = ((1 - sigmoid) * 2.2) + 0.05
    sigmoid_alph = ((1 - sigmoid) * 1.8) + 0.1
    sigmoid_alph = np.clip(sigmoid_alph, 0.1, 1)
    return sigmoid_node, sigmoid_edge, sigmoid_alph


if __name__ == "__main__":

    figure_size = 6
    edge_width = 0.5

    n = 1500

    m_series = np.geomspace(1, n / 2, 12)  # # FIRST HALF

    # m_series = np.geomspace(n / 2, n - 1, 12)  # # SECOND HALF

    for idx, float_m in enumerate(m_series):

        m = int(round(float_m, 0))
        print(idx + 1, f"m: {m}")

        G_BA = nx.barabasi_albert_graph(n, m, seed=0)

        num_edges = len(G_BA.edges())
        diameter = nx.diameter(G_BA)
        density = nx.density(G_BA)

        print()
        print("\tNumber of Edges:", num_edges)
        print("\tDensity:", round(density, 3))
        print("\tDiameter:", diameter)
        print()

        node_sizes, _, edge_alpha = sigmoid_sizes(G_BA.number_of_edges())

        BA_CN, _, BA_CS = BFS_CENSUS(G_BA)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        G_pos = nx.spring_layout(G_BA)

        fig = plt.figure(figsize=(figure_size, figure_size))
        ax = fig.add_subplot(111)
        for edge in G_BA.edges():
            x1, y1 = G_pos[edge[0]]
            x2, y2 = G_pos[edge[1]]
            ax.plot(
                [x1, x2],
                [y1, y2],
                "k-",
                alpha=0.8,
                linewidth=0.3,
            )
        x_values = [pos[0] for pos in G_pos.values()]
        y_values = [pos[1] for pos in G_pos.values()]
        ax.scatter(
            x_values,
            y_values,
            s=3,
            facecolors="k",
            edgecolors="none",
        )
        plt.xticks([])
        plt.yticks([])
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        plt.savefig(
            f"output/BA_m_NL{str(idx+1).zfill(2)}.png",
            format="png",
            dpi=800,
        )
        plt.close()

        # plt.show(block=False)

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

        plt.savefig(
            f"output/BA_m_PS{str(idx+1).zfill(2)}.png",
            format="png",
            dpi=800,
        )
        plt.close()

        # plt.show(block=False)

###############################################################################

# plt.show()
