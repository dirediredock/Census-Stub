from PIL import Image

network_names = [str(i + 1).zfill(2) for i in range(81)]

plot_types = [
    "NL",
    "CN",
    "CE",
    "CS",
    "BN",
    "BE",
    "BS",
    "CN_CE",
    "CE_CS",
    "CN_CS",
]

###############################################################################

image_width, image_height = 600, 600

collage_width, collage_height = 9 * image_width, 9 * image_height
collage = Image.new("RGB", (collage_width, collage_height))

for plot_type in plot_types:
    for i, network_name in enumerate(network_names):
        filename = f"FIG_LocalGlobal/{plot_type}_{network_name}.png"
        img = Image.open(filename)
        img = img.resize((image_width, image_height))

        print(i + 1, "\t", network_name)

        x = (i % 9) * image_width
        y = (i // 9) * image_height

        collage.paste(img, (x, y))

    collage.save(f"FIG_GlobalLocal_{plot_type}.png", dpi=(600, 600))

###############################################################################
