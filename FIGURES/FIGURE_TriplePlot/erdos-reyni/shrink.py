import os
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

types = [
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

for type in types:
    image = Image.open(f"{type}.png")
    resized_image = image.resize((500, 500))
    resized_image.save(f"shrink_{type}.png")
