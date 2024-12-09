import csv

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.sans-serif": "Consolas",
        "font.weight": "bold",
        "font.size": 16,
    }
)

list_filenames = [
    ["network-science", "Empirical"],
    ["bacteria-metabolism", "Empirical"],
    ["college-football", "Empirical"],
    ["human-metabolism", "Empirical"],
    ["bible", "Empirical"],
    ["blogs", "Empirical"],
    ["collaboration", "Empirical"],
    ["crime", "Empirical"],
    ["ecosystem-A", "Empirical"],
    ["ecosystem-B", "Empirical"],
    ["ecosystem-C", "Empirical"],
    ["ecosystem-D", "Empirical"],
    ["email", "Empirical"],
    ["euroroad", "Empirical"],
    ["infectious-expo", "Empirical"],
    ["karate-club", "Empirical"],
    ["languages", "Empirical"],
    ["les-miserables", "Empirical"],
    ["power-grid", "Empirical"],
    ["protein", "Empirical"],
    ["roundworm", "Empirical"],
    ["ten-friends", "Empirical"],
    ["yeast", "Empirical"],
    ["barabasi-albert", "Generated"],
    ["erdos-renyi", "Generated"],
    ["stochastic-block-model", "Generated"],
    ["watts-strogatz", "Generated"],
    ["BA-A", "Generated"],
    ["BA-B", "Generated"],
    ["BA-C", "Generated"],
    ["DGM", "Generated"],
    ["ER-A", "Generated"],
    ["ER-C", "Generated"],
    ["LFR-A", "Generated"],
    ["LFR-B", "Generated"],
    ["percolation-A", "Generated"],
    ["percolation-B", "Generated"],
    ["percolation-C", "Generated"],
    ["percolation-mesh-A", "Generated"],
    ["percolation-mesh-B", "Generated"],
    ["percolation-mesh-C", "Generated"],
    ["random-A", "Generated"],
    ["random-B", "Generated"],
    ["SBM", "Generated"],
    ["WS-A", "Generated"],
    ["WS-C", "Generated"],
    ["WS-D", "Generated"],
    ["desargues", "Geometric"],
    ["dodecahedron", "Geometric"],
    ["example-graph", "Geometric"],
    ["clique-15", "Geometric"],
    ["clique-43", "Geometric"],
    ["equilateral-14", "Geometric"],
    ["equilateral-56", "Geometric"],
    ["frucht", "Geometric"],
    ["grid-14-by-14", "Geometric"],
    ["grid-56-by-56", "Geometric"],
    ["petersen", "Geometric"],
    ["spiderweb-3-100", "Geometric"],
    ["spiderweb-5-200", "Geometric"],
    ["spiderweb-7-300", "Geometric"],
    ["sudoku-2", "Geometric"],
    ["sudoku-3", "Geometric"],
    ["sudoku-4", "Geometric"],
    ["fibonacci-sunflower", "Mesh"],
    ["stanford-bunny", "Mesh"],
    ["camel", "Mesh"],
    ["fox", "Mesh"],
    ["star-14", "Tree"],
    ["star-28", "Tree"],
    ["star-56", "Tree"],
    ["tree-1-8", "Tree"],
    ["tree-14-binomial", "Tree"],
    ["tree-2-7", "Tree"],
    ["tree-3-6", "Tree"],
    ["tree-4-5", "Tree"],
    ["tree-5-4", "Tree"],
    ["tree-7-binomial", "Tree"],
    ["tree-A", "Tree"],
    ["tree-B", "Tree"],
    ["tree-C", "Tree"],
]


def load_CSV(csv_filename):
    with open(csv_filename, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        extract = []
        for row in csv_reader:
            if row[0] == "Number of Nodes":
                extract.append(int(row[1]))
            elif row[0] == "Number of Edges":
                extract.append(int(row[1]))
    return extract


list_N = []
list_E = []
list_type = []

for filename in list_filenames:
    data_CSV = load_CSV(f"DATA/{filename[0]}/statistics.csv")
    list_N.append(data_CSV[0])
    list_E.append(data_CSV[1])
    if filename[1] == "Empirical":
        list_type.append(0)
    elif filename[1] == "Generated":
        list_type.append(1)
    elif filename[1] == "Geometric":
        list_type.append(2)
    elif filename[1] == "Mesh":
        list_type.append(3)
    elif filename[1] == "Tree":
        list_type.append(4)

plt.figure(figsize=(9, 9))

plt.scatter(
    list_N,
    list_E,
    c=list_type,
    cmap="inferno_r",
    s=800,
    alpha=0.7,
    edgecolors="none",
)
plt.scatter(
    list_N,
    list_E,
    c="none",
    s=800,
    edgecolors="k",
)

x = np.linspace(min(list_N), max(list_N), 100)

plt.plot(x, x, "k-", zorder=-1, label="Tree Line")
plt.plot(x, 3 * x, "k:", zorder=-1, label="Planar Horizon")

plt.xlabel("Number of Nodes (log10)")
plt.ylabel("Number of Edges (log10)")

plt.xscale("log")
plt.yscale("log")

plt.tight_layout()

plt.show()
