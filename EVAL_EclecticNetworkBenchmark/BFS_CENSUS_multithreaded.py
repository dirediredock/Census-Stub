import time
from multiprocessing import Pool

import networkx as nx


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


def BFS_CENSUS(G, cores):
    D = nx.to_dict_of_dicts(G)
    Census_Node = []
    Census_Edge = []
    Census_Stub = []
    with Pool(cores) as p:
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

    cores = 11  # number of cores (M3 Macbook Pro)

    G = nx.barabasi_albert_graph(3000, 10, seed=0)

    start_time = time.time()
    Census_Node, Census_Edge, Census_Stub = BFS_CENSUS(G, cores)
    print()
    print(f"Number of cores: 11")
    print(f"Time taken: {time.time() - start_time}")
    print()
