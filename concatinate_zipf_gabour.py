
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
import argparse
import os

path_gabour = 'E:/2 MASTER/Memoire/09-09-2021 (final)/feauteurs/3 Class'
path_zipf = 'E:/2 MASTER/Memoire/09-01-2021'
out = 'E:/2 MASTER/Memoire/09-09-2021 (final)/feauteurs/3 Class'

df_gabour = pd.read_csv(os.path.join(path_gabour,'data_concat_all_no_normaliz_3Class.csv')) #data_concat_non_normaliz
df_zipf = pd.read_csv(os.path.join(path_zipf,'data_concat_all_zipf.csv'))
    # Declare feature vector and target variable
y = df_gabour['target']

gabour = df_gabour.drop(['index', 'img', 'target'], axis=1)
zipf = df_zipf.drop(['index', 'img', 'target'], axis=1)

#df_image_vide = pd.DataFrame(gabour)
#df_image_vide = pd.DataFrame(gabour)
file_csv = pd.concat([df_zipf, gabour],axis=1)
file_csv.to_csv(os.path.join(out,'data_zipf_gabour_no_normaliz_3Class.csv'),index=False)