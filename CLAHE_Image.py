import numpy as np
import cv2
import os
import argparse

from skimage import io
from tqdm import tqdm

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

def get_args():

    parser = argparse.ArgumentParser(description = "Qata_Covid19 Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path_img',type=str,default='E:/2 MASTER/Memoire/09-09-2021 (final)/Test Chest Xray/PNEUMONIA/TEST/CROPED') 
    # arguments for training
    parser.add_argument('--out', type=str, default='E:/2 MASTER/Memoire/09-09-2021 (final)/Test Chest Xray/PNEUMONIA/TEST')

    return parser.parse_args()

def main():
    args = get_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    img_list = get_all_file_paths(args.path_img)
    file_name_img = os.listdir(args.path_img)
    i = 0
    for file in tqdm(img_list):
        img = cv2.imread(file, 1)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_img)
        ########### CLAHE ###########
        #Apply CLAHE to L channel
        clahe = cv2.createCLAHE ( clipLimit = 3.0 , tileGridSize = ( 8,8 ) )
        clahe_img = clahe.apply (l)
        updated_lab_img2 = cv2.merge((clahe_img, a, b))
        #Convert LAB image back to color ( RGB )
        CLAHE_img = cv2.cvtColor ( updated_lab_img2 , cv2.COLOR_LAB2BGR )
        cv2.imwrite(os.path.join(args.out,file_name_img[i]),CLAHE_img)
        i = i + 1
    
if __name__ == '__main__':
    main()


#j = 0
#for file in tqdm(mask_list):
#    img = cv2.imread(file, 1)
#    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#    l, a, b = cv2.split(lab_img)
    ########### CLAHE ###########
    #Apply CLAHE to L channel
#    clahe = cv2.createCLAHE ( clipLimit = 3.0 , tileGridSize = ( 8,8 ) )
#    clahe_img = clahe.apply (l)
#    updated_lab_img2 = cv2.merge((clahe_img, a, b))
    #Convert LAB image back to color ( RGB )
#    CLAHE_img = cv2.cvtColor ( updated_lab_img2 , cv2.COLOR_LAB2BGR )
#    cv2.imwrite(os.path.join(out_msk,file_name_mask[j]),CLAHE_img)
#    j = j + 1
#print(img_list)