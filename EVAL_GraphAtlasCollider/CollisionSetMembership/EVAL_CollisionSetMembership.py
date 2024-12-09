import csv
import json
import os

import networkx as nx

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

###############################################################################

graph_order = 3

###############################################################################

JSON_file_CN = f"../TGC_collisions/CN/Order{graph_order}_atlas_key_CN.json"
JSON_file_CE = f"../TGC_collisions/CE/Order{graph_order}_atlas_key_CE.json"
JSON_file_CS = f"../TGC_collisions/CS/Order{graph_order}_atlas_key_CS.json"

with open(JSON_file_CN, "r") as json_file:
    data_CN = json.load(json_file)
json_file.close()

with open(JSON_file_CE, "r") as json_file:
    data_CE = json.load(json_file)
json_file.close()

with open(JSON_file_CS, "r") as json_file:
    data_CS = json.load(json_file)
json_file.close()

collisions_CN = []
collisions_CE = []
collisions_CS = []

for key, values in data_CN.items():
    for value in values:
        collisions_CN.append(value)

for key, values in data_CE.items():
    for value in values:
        collisions_CE.append(value)

for key, values in data_CS.items():
    for value in values:
        collisions_CS.append(value)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

strings_Graph6 = []
graph_objects = []
graph_idx = []

with open(
    "../Graph_Atlas/order_" + str(graph_order) + ".g6",
    "rb",
) as graph_file:
    count = 0
    idx = 0
    for graph_compressed in graph_file.readlines():
        graph = nx.from_graph6_bytes(graph_compressed.rstrip())
        if nx.is_connected(graph):
            if count % 100 == 0:
                print(count, "loaded")
            count += 1
            graph_idx.append(idx)
            strings_Graph6.append(str(graph_compressed.decode())[:-1])
            graph_objects.append(graph)
        idx += 1
graph_file.close()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

trio_sets = []

with open(f"CollisionSet_CSV/DATA_{graph_order}.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "idx",
            "Graph6",
            "edges",
            "CN",
            "CE",
            "CS",
        ]
    )

    for i, G in enumerate(graph_objects):
        if i % 100 == 0:
            print(i, "analyzed")

        idx = graph_idx[i]

        E = G.number_of_edges()

        trio = [0, 0, 0]

        if idx in collisions_CN:
            trio[0] = 1
        if idx in collisions_CE:
            trio[1] = 1
        if idx in collisions_CS:
            trio[2] = 1

        if trio not in trio_sets:
            trio_sets.append(trio)

        writer.writerow(
            [
                idx,
                strings_Graph6[i],
                E,
                trio[0],
                trio[1],
                trio[2],
            ]
        )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

print()
print(trio_sets)
print()

###############################################################################
