import json
from copy import deepcopy
from csv import writer
from math import comb as N_Choose_2

import networkx as nx
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


def key_Census_of(Census):
    Census_string = ""
    for row in sorted(Census):
        row_string = ""
        for value in row:
            row_string += str(int(value)) + "-"
        Census_string += row_string + "|"
    return Census_string


def key_BMatrix_of(BMatrix):
    BMatrix_string = ""
    for row in BMatrix:
        row_string = ""
        for value in row:
            row_string += str(int(value)) + "-"
        BMatrix_string += row_string + "|"
    return BMatrix_string


def Filter_Collisions(atlas_keys):
    dict_collisions = {}
    for idx, signal_key in enumerate(atlas_keys):
        if idx not in atlas_disconnected:
            if signal_key in dict_collisions:
                dict_collisions[signal_key].append(idx)
            else:
                dict_collisions[signal_key] = [idx]
    filtered_collisions = {}
    for key, value in dict_collisions.items():
        if len(value) >= 2:
            filtered_collisions[key] = value
    return filtered_collisions


def BFS_BMatrix(G):
    diameter = 1000
    array = zeroes_array((diameter + 1, G.number_of_nodes()))
    max_path = 1
    adjencent_nodes = G.adj
    for starting_node in G.nodes():
        nodes_visited = {starting_node: 0}
        search_queue = [starting_node]
        count = 1
        while search_queue:
            next_depth = []
            extend = next_depth.extend
            for n in search_queue:
                l = [i for i in adjencent_nodes[n] if i not in nodes_visited]
                extend(l)
                for j in l:
                    nodes_visited[j] = count
            search_queue = next_depth
            count += 1
        node_distances = nodes_visited.values()
        max_node_distances = max(node_distances)
        curr_max_path = max_node_distances
        if curr_max_path > max_path:
            max_path = curr_max_path
        dict_distribution = dict.fromkeys(node_distances, 0)
        for count in node_distances:
            dict_distribution[count] += 1
        for shell, count in dict_distribution.items():
            array[shell][count] += 1
        max_shell = diameter
        while max_shell > max_node_distances:
            array[max_shell][0] += 1
            max_shell -= 1
    BMatrix_08 = array[: max_path + 1, :]
    trim = np.where(BMatrix_08 != 0)
    BMatrix_08 = BMatrix_08[
        0 : max(trim[0]) + 1,
        0 : max(trim[1]) + 1,
    ]
    return BMatrix_08


def zeroes_array(shape, data_type=None):
    try:
        return np.zeros(shape, dtype=data_type)
    except TypeError:
        return np.zeros(shape, typecode="fd")


###############################################################################

graph_order = 8

###############################################################################

with open(
    "Graph_Atlas/order_" + str(graph_order) + ".g6",
    "rb",
) as graph_file:
    graph_objects = []
    count = 0
    for graph_compressed in graph_file.readlines():
        if count % 100 == 0:
            print(count, "loaded")
        count += 1
        graph = nx.from_graph6_bytes(graph_compressed.rstrip())
        graph_objects.append(graph)
graph_file.close()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

atlas_idx = []
atlas_disconnected = []

atlas_key_Census_Node = []
atlas_key_Census_Edge = []
atlas_key_Census_Stub = []

atlas_key_BMatrix_Node = []
atlas_key_BMatrix_Edge = []
atlas_key_BMatrix_Stub = []

atlas_key_BMatrix_08 = []

atlas_key_diameter = []
atlas_key_degree = []

for i, G in enumerate(graph_objects):
    if i % 100 == 0:
        print(i, "analyzed")
    atlas_idx.append(i)
    if not nx.is_connected(G):
        atlas_disconnected.append(i)

        atlas_key_Census_Node.append("")
        atlas_key_Census_Edge.append("")
        atlas_key_Census_Stub.append("")
        atlas_key_BMatrix_Node.append("")
        atlas_key_BMatrix_Edge.append("")
        atlas_key_BMatrix_Stub.append("")
        atlas_key_BMatrix_08.append("")
        atlas_key_diameter.append("")
        atlas_key_degree.append("")

    else:
        Census_of_Nodes, Census_of_Edges, Census_of_Stubs = BFS_Census(G)

        BMatrix_08 = BFS_BMatrix(G)

        key_Census_of_Node = key_Census_of(Census_of_Nodes)
        key_Census_of_Edge = key_Census_of(Census_of_Edges)
        key_Census_of_Stub = key_Census_of(Census_of_Stubs)

        BMatrix_of_Node = BMatrix_of_Census(Census_of_Nodes)
        BMatrix_of_Edge = BMatrix_of_Census(Census_of_Edges)
        BMatrix_of_Stub = BMatrix_of_Census(Census_of_Stubs)

        key_BMatrix_of_Node = key_BMatrix_of(BMatrix_of_Node)
        key_BMatrix_of_Edge = key_BMatrix_of(BMatrix_of_Edge)
        key_BMatrix_of_Stub = key_BMatrix_of(BMatrix_of_Stub)

        key_BMatrix_08 = key_BMatrix_of(BMatrix_08)

        atlas_key_Census_Node.append(key_Census_of_Node)
        atlas_key_Census_Edge.append(key_Census_of_Edge)
        atlas_key_Census_Stub.append(key_Census_of_Stub)

        atlas_key_BMatrix_Node.append(key_BMatrix_of_Node)
        atlas_key_BMatrix_Edge.append(key_BMatrix_of_Edge)
        atlas_key_BMatrix_Stub.append(key_BMatrix_of_Stub)

        atlas_key_BMatrix_08.append(key_BMatrix_08)

        int_diameter = nx.diameter(G)
        atlas_key_diameter.append(str(int_diameter).zfill(5))

        degree_dist = sorted([row[:1] for row in Census_of_Nodes])
        key_degree = key_Census_of(degree_dist)
        atlas_key_degree.append(key_degree)

###############################################################################

with open(
    "TGC_collisions/disconnected_graphs_idx/Order"
    + str(graph_order)
    + "_disconnected.csv",
    "w",
    newline="",
) as csv_file:
    csv_writer = writer(csv_file)
    for idx in atlas_disconnected:
        csv_writer.writerow([idx])
csv_file.close()

print()

###############################################################################

collisions_dict = Filter_Collisions(atlas_key_Census_Node)

final_result = 0
for key, value in collisions_dict.items():
    final_result += N_Choose_2(len(value), 2)

with open(
    "TGC_collisions/CN/Order" + str(graph_order) + "_atlas_key_CN.json",
    "w",
) as outfile:
    json.dump(collisions_dict, outfile)
outfile.close()

result_CN = final_result

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

collisions_dict = Filter_Collisions(atlas_key_Census_Edge)

final_result = 0
for key, value in collisions_dict.items():
    final_result += N_Choose_2(len(value), 2)

with open(
    "TGC_collisions/CE/Order" + str(graph_order) + "_atlas_key_CE.json",
    "w",
) as outfile:
    json.dump(collisions_dict, outfile)
outfile.close()

result_CE = final_result

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

collisions_dict = Filter_Collisions(atlas_key_Census_Stub)

final_result = 0
for key, value in collisions_dict.items():
    final_result += N_Choose_2(len(value), 2)

with open(
    "TGC_collisions/CS/Order" + str(graph_order) + "_atlas_key_CS.json",
    "w",
) as outfile:
    json.dump(collisions_dict, outfile)
outfile.close()

result_CS = final_result

###############################################################################

collisions_dict = Filter_Collisions(atlas_key_BMatrix_Node)

final_result = 0
for key, value in collisions_dict.items():
    final_result += N_Choose_2(len(value), 2)

with open(
    "TGC_collisions/BN/Order" + str(graph_order) + "_atlas_key_BN.json",
    "w",
) as outfile:
    json.dump(collisions_dict, outfile)
outfile.close()

result_BN = final_result

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

collisions_dict = Filter_Collisions(atlas_key_BMatrix_Edge)

final_result = 0
for key, value in collisions_dict.items():
    final_result += N_Choose_2(len(value), 2)

with open(
    "TGC_collisions/BE/Order" + str(graph_order) + "_atlas_key_BE.json",
    "w",
) as outfile:
    json.dump(collisions_dict, outfile)
outfile.close()

result_BE = final_result

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

collisions_dict = Filter_Collisions(atlas_key_BMatrix_Stub)

final_result = 0
for key, value in collisions_dict.items():
    final_result += N_Choose_2(len(value), 2)

with open(
    "TGC_collisions/BS/Order" + str(graph_order) + "_atlas_key_BS.json",
    "w",
) as outfile:
    json.dump(collisions_dict, outfile)
outfile.close()

result_BS = final_result

###############################################################################

collisions_dict = Filter_Collisions(atlas_key_BMatrix_08)

final_result = 0
for key, value in collisions_dict.items():
    final_result += N_Choose_2(len(value), 2)

with open(
    "TGC_collisions/BN_check/Order" + str(graph_order) + "_atlas_key_BN_check.json",
    "w",
) as outfile:
    json.dump(collisions_dict, outfile)
outfile.close()

result_BN_check = final_result

###############################################################################

collisions_dict = Filter_Collisions(atlas_key_diameter)

final_result = 0
for key, value in collisions_dict.items():
    final_result += N_Choose_2(len(value), 2)

with open(
    "TGC_collisions/diameter/Order" + str(graph_order) + "_atlas_key_diameter.json",
    "w",
) as outfile:
    json.dump(collisions_dict, outfile)
outfile.close()

result_diameter = final_result

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

collisions_dict = Filter_Collisions(atlas_key_degree)

final_result = 0
for key, value in collisions_dict.items():
    final_result += N_Choose_2(len(value), 2)

with open(
    "TGC_collisions/degree_distribution/Order"
    + str(graph_order)
    + "_atlas_key_degree_distribution.json",
    "w",
) as outfile:
    json.dump(collisions_dict, outfile)
outfile.close()

result_degree = final_result

###############################################################################

print()
print("Census of Nodes\t\t", result_CN)
print("Census of Edges\t\t", result_CE)
print("Census of Stubs\t\t", result_CS)
print()
print("BMatrix of Nodes\t", result_BN)
print("BMatrix of Edges\t", result_BE)
print("BMatrix of Stubs\t", result_BS)
print()
print("BFS_BMatrix\t\t", result_BN_check)
print()
print("Diameter\t\t", result_diameter)
print("Degree\t\t\t", result_degree)
print()

###############################################################################
