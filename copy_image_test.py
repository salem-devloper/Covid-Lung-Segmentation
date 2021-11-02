
from posix import listdir
from shutil import copyfile

import os
import argparse
import sys

def get_args():

    parser = argparse.ArgumentParser(description = "Qata_Covid19 Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path_img',type=str,default='E:/2 MASTER/Memoire/07-06-2021 (croped)/test/pneumonia') 
    parser.add_argument('--path_img_test', type=str, default='image_test')
    # arguments for training
    parser.add_argument('--out', type=str, default='./data')

    return parser.parse_args()

def main():
    args = get_args()

    # copy image file 
    os.mkdir(args.out)
    os.mkdir(os.path.join(args.out,'folder_img'))
    list_img = os.listdir(args.path_img)
    for img_name in list_img:
        copyfile(os.path.join(args.path_img,os.path.basename(img_name)),
            os.path.join(args.out,'folder_img',os.path.basename(img_name)))

    # copy and rename image test
    list_img_test = os.listdir(args.path_img_test)
    image_name = os.path.basename(list_img_test[0])
    list1 = image_name.split('.')

    for i in list1:
        if (i == 'png' or i == 'jpeg' or i == 'jpg'):
            format_img = i
            find = True
    
    if not find:
        print("this format Not supported")
        sys.exit()

    for img_name in list_img_test:
        copyfile(os.path.join(args.path_img_test,os.path.basename(img_name)),
            os.path.join(args.out,'folder_img','image_test.' + format_img))

if __name__ == '__main__':
    
    main()