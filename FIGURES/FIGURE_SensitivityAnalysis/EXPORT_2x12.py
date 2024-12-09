from PIL import Image

image_width, image_height = 500, 500
collage_width, collage_height = 12 * image_width, 2 * image_height

collage = Image.new("RGB", (collage_width, collage_height))

# figure_type = "ER_p"
# figure_type = "BA_m"
# figure_type = "WS_p"

figure_type = "Loop"

for i in range(12):
    filename = f"output/{figure_type}_NL{str(i+1).zfill(2)}.png"
    img = Image.open(filename)
    img = img.resize((image_width, image_height))

    x = (i % 12) * image_width
    y = 0

    collage.paste(img, (x, y))

for i in range(12):
    filename = f"output/{figure_type}_PS{str(i+1).zfill(2)}.png"
    img = Image.open(filename)
    img = img.resize((image_width, image_height))

    x = (i % 12) * image_width
    y = image_height

    collage.paste(img, (x, y))

collage.save(f"{figure_type}.png", dpi=(600, 600))
