from PIL import Image
from tqdm import tqdm
import os

def get_all_file_paths(directory):

    # initializing empty file paths list
    file_paths = []

    # crawling through directory and subdirectories
    print("get all file paths")
    for root, directories, files in tqdm(os.walk(directory)):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    # returning all file paths
    return file_paths

path = "E:/2 MASTER/Memoire/09-09-2021 (final)/Test Chest Xray/PNEUMONIA/TEST/ORIGINAL"
out = "E:/2 MASTER/Memoire/09-09-2021 (final)/Test Chest Xray/PNEUMONIA/TEST/RESIZE"

img_list = get_all_file_paths(path)
file_name_img = os.listdir(path)
i = 0

for file in tqdm(img_list):
    img = Image.open(file) # image extension *.png,*.jpg
    new_width  = 299
    new_height = 299
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img.save(os.path.join(out,file_name_img[i])) # format may what you want *.png, *jpg, *.gif
    i = i + 1