import os
import numpy as np
import pandas as pd
import argparse
import pickle

from PIL import Image
from shutil import copyfile
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from csv import reader,writer
from tqdm import tqdm
from process_data import process_image
from create_annotation import create_annotation

def get_args():

    parser = argparse.ArgumentParser(description = "Qata_Covid19 Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path_img',type=str,default='E:/2 MASTER/Memoire/07-06-2021 (croped)/test/covid') #covid pneumonia
    parser.add_argument('--name_img', type=str, default='image_test')
    parser.add_argument('--folder_name', type=str, default='COVID19')
    # arguments for training
    parser.add_argument('--loader_model_svm', type = str , default = 'E:/2 MASTER/Memoire/07-06-2021 (croped)/linear_svc_model_normaliz_version01.sav')
    parser.add_argument('--data_not_normaliz', type=str, default='E:/2 MASTER/Memoire/07-06-2021 (croped)/data_concat_non_normaliz.csv')
    parser.add_argument('--out', type=str, default='E:/2 MASTER/Memoire/07-12-2021 (file csv)/data csv')

    return parser.parse_args()

def main_res(out, data_not_normaliz, path_img, loader_model_svm):

    #args = get_args()

    # remove file csv
    if os.path.exists(os.path.join(out,'data.csv')):
        os.remove(os.path.join(out,'data.csv'))
    if os.path.exists(os.path.join(out,'data_add.csv')):
        os.remove(os.path.join(out,'data_add.csv'))
    if os.path.exists(os.path.join(out,'data_normalization.csv')):
        os.remove(os.path.join(out,'data_normalization.csv'))

    # create path out save
    if not os.path.exists(out):
        os.mkdir(out)

    # create data frame vide to image input
    df_image_vide = pd.DataFrame(columns = ['index','img','target','0','1','2','3','4','5','6','7'])
    file_csv = pd.concat([df_image_vide])

    file_csv.to_csv(os.path.join(out,'data.csv'),index=False)

    # copy data no normalization to data frame image input
    with open(data_not_normaliz, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Iterate over each row in the csv using reader object
        with open(os.path.join(out,'data.csv'), 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            if header != None:
                for row in tqdm(csv_reader):
                    csv_writer.writerow(row)

    # get image process
    img_list = os.listdir(path_img)
    data = []
    for img_name in tqdm(img_list):
    #if (os.path.basename(img_name) == 'image_test.png') or (os.path.basename(img_name) == 'image_test.jpeg') or (os.path.basename(img_name) == 'image_test.jpg') :
    #print('image find name: ',os.path.basename(img_name))
        image_name = os.path.basename(img_name)
        list1 = image_name.split('.')
        for i in list1:
            if i == 'image_test':
                img = np.array(Image.open(os.path.join(path_img,os.path.basename(img_name))).convert('L'))
                break
        # Process image (get features)
    features = process_image(img)        
    data.append(features)
    #break

    #df = create_annotation(args.path_img, 0)
    
    feature = pd.DataFrame(data)
    #feature = pd.concat([df,feature], axis=1)
    #feature.to_csv(os.path.join(args.out,'data_process.csv'),index=False)

    

    # create annotation and effect new row target
    annotation = pd.DataFrame(columns = ['index','img','target'])
    new_row = {'index':17227, 'img':'image_test', 'target':0}
    #append row to the dataframe
    annotation = annotation.append(new_row, ignore_index=True)

    # concatinate annotation end feature to final df
    final_df = pd.concat([annotation,feature],axis=1)
    final_df.to_csv(os.path.join(out,'data_add.csv'),index=False)

    # add data_add row to data final
    with open(os.path.join(out,'data_add.csv'), 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Iterate over each row in the csv using reader object
        with open(os.path.join(out,'data.csv'), 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            if header != None:
                for row in tqdm(csv_reader):
                    csv_writer.writerow(row)

    # normalization data final and effect to data_normalization
    df = pd.read_csv(os.path.join(out,'data.csv'))
    df1 = df.drop(['0','1','2','3','4','5','6','7'], axis=1)
    x_array = df.drop(['index','img','target'], axis=1)
    #info = df['img']
    names = x_array.columns
    scaler = MinMaxScaler()
    scaler.fit(x_array)
    d = scaler.transform(x_array)
    scaled_df = pd.DataFrame(d, columns=names)
    final_df = pd.concat([scaled_df],axis=1)
    final_df.to_csv(os.path.join(out,'data_normalization.csv'),index=False)
    print("Normalization Done")

    # find row image_test normalization
    with open(os.path.join(out,'data_normalization.csv'), 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        i = 0
        N = 17227
        for row in csv_reader:
            if i == N:
                print("This is the line.")
                print(row)
                features_image_test = row
                break
            i += 1

    #df = pd.read_csv(os.path.join(args.out,'data_normalization.csv'))
    #te = df.drop(['index','img','target'], axis=1)
    #x = np.array([features_image_test])
    #features_image_test = x.astype(np.float)
    #print(x)
    #features_image_test = np.around(x,9)
    #features_image_test = np.set_printoptions(formatter={'float': lambda x: "{0:0.9f}".format([x])})
    #print("features_image_test : ",features_image_test)
    #scaled_df = pd.DataFrame(features_image_test)
    #final_df = pd.concat([scaled_df],axis=1)
    #final_df.to_csv(os.path.join(args.out,'features_image_test.csv'),index=False)

    # loaded model SVM to classification image
    loaded_model = pickle.load(open(loader_model_svm, 'rb'))
    #print(features_image_test)
    result = loaded_model.predict([features_image_test])
    print(result)
    #scaled_df = pd.DataFrame([result])
    #final_df = pd.concat([scaled_df],axis=0)
    #final_df.to_csv(os.path.join(args.out,'data_result.csv'),index=False)
    if result == [0]:    
        print("image input is NORMAL")
    if result == [1]:    
        print("image input is COVID")
    if result == [2]:    
        print("image input is PNEUMONIA")
    
    return result

if __name__ == '__main__':
    args = get_args()
    main_res(args.out, args.data_not_normaliz, args.path_img, args.loader_model_svm)