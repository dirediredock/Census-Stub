import networkx as nx
import pickle
import sys
import csv
from collections import Counter
from copy import deepcopy
import numpy as np


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
    matrix_X, matrix_Y = get_BMatrix_size(census_array)
    BMatrix_aggregate = [[0 for _ in range(matrix_Y)] for _ in range(matrix_X)]
    for signal in census_array:
        if len(signal) < matrix_X:
            signal += [0] * (matrix_X - len(signal))
        for row, col in enumerate(signal):
            BMatrix_aggregate[row][col] += 1
    return BMatrix_aggregate


def string_edgelist(edgelist):
    edgelist_string = ""
    for row in edgelist:
        edgelist_string += str(int(row[0])) + "-" + str(int(row[1])) + "|"
    return edgelist_string


def string_DD(sequence):
    degree_string = ""
    for value in sequence:
        degree_string += str(int(value)) + "-"
    return degree_string


def string_Census(Census):
    Census_string = ""
    for row in sorted(Census):
        row_string = ""
        for value in row:
            row_string += str(int(value)) + "-"
        Census_string += row_string + "|"
    return Census_string


def string_BMatrix(BMatrix):
    BMatrix_string = ""
    for row in BMatrix:
        row_string = ""
        for value in row:
            row_string += str(int(value)) + "-"
        BMatrix_string += row_string + "|"
    return BMatrix_string


###############################################################################

up_to_Graph_Atlas_order = 10

###############################################################################

idx_graph_order = []
bytes_graph6 = []
bytes_edgelist = []
bytes_diameter = []
bytes_degree_distribution = []
bytes_CN = []
bytes_CE = []
bytes_CS = []
bytes_BN = []
bytes_BE = []
bytes_BS = []

for graph_order in range(3, up_to_Graph_Atlas_order + 1):
    with open(
        "Graph_Atlas/order_" + str(graph_order) + ".g6",
        "rb",
    ) as graph_file:
        graph6_string = []
        graph_objects = []
        count = 0
        for graph_compressed in graph_file.readlines():
            graph = nx.from_graph6_bytes(graph_compressed.rstrip())
            if nx.is_connected(graph):
                if count % 100 == 0:
                    print(count, "loaded")
                count += 1
                graph6_string.append(str(graph_compressed.decode())[:-1])
                graph_objects.append(graph)
    graph_file.close()

    for i, G in enumerate(graph_objects):
        if i % 100 == 0:
            print(i, "analyzed")

        idx_graph_order.append(graph_order)
        degre_distribution = sorted(list(dict(G.degree()).values()))

        Census_of_Nodes, Census_of_Edges, Census_of_Stubs = BFS_Census(G)
        BMatrix_of_Node = BMatrix_of_Census(Census_of_Nodes)
        BMatrix_of_Edge = BMatrix_of_Census(Census_of_Edges)
        BMatrix_of_Stub = BMatrix_of_Census(Census_of_Stubs)

        dump_graph6 = pickle.dumps(str(graph6_string[i]))
        dump_edgelist = pickle.dumps(string_edgelist(G.edges()))
        dump_diameter = pickle.dumps(str(int(nx.diameter(G))))
        dump_DD = pickle.dumps(string_DD(degre_distribution))
        dump_CN = pickle.dumps(string_Census(Census_of_Nodes))
        dump_CE = pickle.dumps(string_Census(Census_of_Edges))
        dump_CS = pickle.dumps(string_Census(Census_of_Stubs))
        dump_BN = pickle.dumps(string_BMatrix(BMatrix_of_Node))
        dump_BE = pickle.dumps(string_BMatrix(BMatrix_of_Edge))
        dump_BS = pickle.dumps(string_BMatrix(BMatrix_of_Stub))

        bytes_graph6.append(sys.getsizeof(dump_graph6))
        bytes_edgelist.append(sys.getsizeof(dump_edgelist))
        bytes_diameter.append(sys.getsizeof(dump_diameter))
        bytes_degree_distribution.append(sys.getsizeof(dump_DD))
        bytes_CN.append(sys.getsizeof(dump_CN))
        bytes_CE.append(sys.getsizeof(dump_CE))
        bytes_CS.append(sys.getsizeof(dump_CS))
        bytes_BN.append(sys.getsizeof(dump_BN))
        bytes_BE.append(sys.getsizeof(dump_BE))
        bytes_BS.append(sys.getsizeof(dump_BS))

###############################################################################

dict_graph6 = Counter(bytes_graph6)
dict_edgelist = Counter(bytes_edgelist)
dict_diameter = Counter(bytes_diameter)
dict_degree_distribution = Counter(bytes_degree_distribution)
dict_CN = Counter(bytes_CN)
dict_CE = Counter(bytes_CE)
dict_CS = Counter(bytes_CS)
dict_BN = Counter(bytes_BN)
dict_BE = Counter(bytes_BE)
dict_BS = Counter(bytes_BS)

with open("TGC_bytes/graph6.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["byte", "frequency"])
    for key, value in dict_graph6.items():
        writer.writerow([key, value])

with open("TGC_bytes/edgelist.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["byte", "frequency"])
    for key, value in dict_edgelist.items():
        writer.writerow([key, value])

with open("TGC_bytes/diameter.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["byte", "frequency"])
    for key, value in dict_diameter.items():
        writer.writerow([key, value])

with open("TGC_bytes/degree_distribution.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["byte", "frequency"])
    for key, value in dict_degree_distribution.items():
        writer.writerow([key, value])

with open("TGC_bytes/CN.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["byte", "frequency"])
    for key, value in dict_CN.items():
        writer.writerow([key, value])

with open("TGC_bytes/CE.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["byte", "frequency"])
    for key, value in dict_CE.items():
        writer.writerow([key, value])

with open("TGC_bytes/CS.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["byte", "frequency"])
    for key, value in dict_CS.items():
        writer.writerow([key, value])

with open("TGC_bytes/BN.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["byte", "frequency"])
    for key, value in dict_BN.items():
        writer.writerow([key, value])

with open("TGC_bytes/BE.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["byte", "frequency"])
    for key, value in dict_BE.items():
        writer.writerow([key, value])

with open("TGC_bytes/BS.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["byte", "frequency"])
    for key, value in dict_BS.items():
        writer.writerow([key, value])

###############################################################################
