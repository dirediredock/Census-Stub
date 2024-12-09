from PIL import Image

network_names = [
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

###############################################################################

plot_type = [
    "NL",
    # "CN",
    # "CE",
    # "CS",
    # "BN",
    # "BE",
    # "BS",
    # "CN_CE",
    # "CE_CS",
    # "CN_CS",
]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

image_width, image_height = 600, 600

collage_width, collage_height = 9 * image_width, 9 * image_height
collage = Image.new("RGB", (collage_width, collage_height))

for i, network_name in enumerate(network_names):
    filename = f"FIG_VisualEncodingPipeline/{network_name}_{plot_type[0]}.png"
    img = Image.open(filename)
    img = img.resize((image_width, image_height))

    print(i + 1, "\t", network_name)

    x = (i % 9) * image_width
    y = (i // 9) * image_height

    collage.paste(img, (x, y))

collage.save(f"FIG_Collage_{plot_type[0]}.png", dpi=(600, 600))

###############################################################################
