import hickle
import numpy as np
import os

from PIL import Image

if __name__ == "__main__":
    d = hickle.load("data/September_1_total_non_overlap.hkl")

    GROUPS = {
        'NORMAL': 0,
        'Echinocyte': 1,
        'Dacrocyte': 2,
        'Schistocyte': 3,
        'Elliptocyte': 4,
        'Acanthocyte': 5,
        'Target cell': 6,
        'Stomatocyte': 7,
        'Spherocyte': 8,
        'Overlap': 9
    }
    LABELS = {v:k for k, v in GROUPS.items()}

    counter = 0
    for label, image_array, z in zip(d['y'], d['X'], d['pk']):
        label = label.lower()
        image_array = np.array(image_array)

        sub_dir = os.path.join("images", label)

        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        image_file_name = "{}.jpg".format(counter)
        image_file_path = os.path.join(sub_dir, image_file_name)

        image = Image.fromarray(image_array)
        image.save(image_file_path)
        counter += 1
