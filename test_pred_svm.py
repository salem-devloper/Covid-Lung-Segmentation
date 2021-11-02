import pickle
import os
import pandas as pd

out = 'E:/2 MASTER/Memoire/08-03-2021/AppHeroku/model'
path = 'E:/2 MASTER/Memoire/09-09-2021 (final)/Test Chest Xray/NORMAL'
filename = os.path.join(out, 'svc_rbf_c100_model_normaliz_3Class.sav')

df = pd.read_csv(os.path.join(path,'data_normaliz_all.csv')) #data_concat_non_normaliz

# Declare feature vector and target variable

X = df.drop(['index', 'img', 'target'], axis=1)
#print(X)

y = df['target']

loaded_model = pickle.load(open(filename, 'rb'))

tes = loaded_model.predict(X)
data = []
for i in tes:
    data.append([i])
df_image_vide = pd.DataFrame(data, columns = ['target'])
file_csv = pd.concat([df_image_vide])
file_csv.to_csv(os.path.join(path,'data_pred_normaliz.csv'),index=False)
print("save file csv")
