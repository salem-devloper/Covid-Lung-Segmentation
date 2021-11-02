# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
import argparse
import os
#matplotlib inline

# Any results you write to the current directory are saved as output.
def get_args():

    parser = argparse.ArgumentParser(description = "U-Net for Lung Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path', type=str, default = 'E:/2 MASTER/Memoire/09-09-2021 (final)/feauteurs/3 Class')
    parser.add_argument('--out', type=str, default = 'E:/2 MASTER/Memoire/09-09-2021 (final)/feauteurs/3 Class')
    return parser.parse_args()

def main():

#data = '/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv'

    args = get_args()

    df = pd.read_csv(os.path.join(args.path,'data_zipf_gabour_normaliz_all.csv')) #data_concat_non_normaliz

    # Declare feature vector and target variable

    X = df.drop(['index', 'img', 'target'], axis=1)
    #print(X)

    y = df['target']
    #print(y)
    # split X and y into training and testing sets

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


    #cols = X_train.columns

    from sklearn.preprocessing import StandardScaler

    #scaler = StandardScaler()

    #X_train = scaler.fit_transform(X_train)

    #X_test = scaler.transform(X_test)

    #X_train = pd.DataFrame(X_train, columns=[cols])

    #X_test = pd.DataFrame(X_test, columns=[cols])
    #print("scaler x_train")
    #print(X_train)
    #print("scaler x_test")
    #print(X_test)
    
    #Run SVM with default hyperparameters 
    #Table of Contents
    #Default hyperparameter means C=1.0, kernel=rbf and gamma=auto among other parameters.

    # import SVC classifier
    from sklearn.svm import SVC


    # import metrics to compute accuracy
    from sklearn.metrics import accuracy_score


    # instantiate classifier with default hyperparameters
    svc_hyperpar=SVC() 


    # fit classifier to training set
    svc_hyperpar.fit(X_train,y_train)


    # make predictions on test set
    y_pred=svc_hyperpar.predict(X_test)

    #print("test pridect normaliz 2")
    #print(svc_hyperpar.predict([[-2.33E-06,	2.35E-05,	6.17E-07,	1.75E-06,	1.03E-05,	1,	0,	-2.20E-07]]))
    #print(svc_hyperpar.predict([[-2.02E-05,	0.000168,	7.43E-06,	1.57E-05,	7.83E-05,	0.999999982,	0,	-7.29E-06]]))
    #print(svc_hyperpar.predict([[-1.91E-05,	0.000161029,	6.05E-06,	1.70E-05,	8.54E-05,	0.999999983,	0,	-7.39E-06]]))

    #print("test pridect normaliz svc_hyperpar")
    #print(svc_hyperpar.predict([[0.404695112, 0.556220642, 0.435032654, 0.95375712, 0.616367614, 0.140135251, 0, 0.801199482]]))
    #print(svc_hyperpar.predict([[0.572075354,	0.425227225,	0.196546918,	0.873134995,	0.922964603,	0.138429271,	0.159571425,	0.449790328]]))
    #print(svc_hyperpar.predict([[0.658484677,	0.443266785,	0.494248695,	0.944447685,	0.579502115,	0.503946736,	0.162076535,	0.779742237]]))

    #print("test pridect")
    #print(svc_hyperpar.predict([[-1.317997098,	13.34376144,	0.349806249,	0.991162002,	5.823415756,	566667.5,	0,	-0.12453936]]))
    #print(svc_hyperpar.predict([[-1.138931751,	9.900605202,	0.517118871,	0.968591928,	4.700357437,	64236,	0,	-0.32606703]]))
    #print(svc_hyperpar.predict([[-1.156782269,	9.676625252,	0.403339744,	0.959918499,	4.784446239,	58879,	0,	-0.407173276]]))


    # compute and print accuracy score
    print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

    #Run SVM with rbf kernel and C=100.0
    #We have seen that there are outliers in our dataset. 
    #So, we should increase the value of C as higher C means fewer outliers. 
    #So, I will run SVM with kernel=rbf and C=100.0.

    # instantiate classifier with rbf kernel and C=100
    svc_rbf_c100=SVC(C=100.0) 


    # fit classifier to training set
    svc_rbf_c100.fit(X_train,y_train)


    # make predictions on test set
    y_pred=svc_rbf_c100.predict(X_test)

    #print("test pridect normaliz svc_rbf_c100")
    #print(svc_rbf_c100.predict([[0.404695112, 0.556220642, 0.435032654, 0.95375712, 0.616367614, 0.140135251, 0, 0.801199482]]))
    #print(svc_rbf_c100.predict([[0.572075354,	0.425227225,	0.196546918,	0.873134995,	0.922964603,	0.138429271,	0.159571425,	0.449790328]]))
    #print(svc_rbf_c100.predict([[0.658484677,	0.443266785,	0.494248695,	0.944447685,	0.579502115,	0.503946736,	0.162076535,	0.779742237]]))

    # compute and print accuracy score
    print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

    #Run SVM with rbf kernel and C=1000.0

    # instantiate classifier with rbf kernel and C=1000
    svc_rbf_c1000 = SVC(C=1000.0) 


    # fit classifier to training set
    svc_rbf_c1000.fit(X_train,y_train)


    # make predictions on test set
    y_pred=svc_rbf_c1000.predict(X_test)
    
    #print("test pridect normaliz svc_rbf_c1000")
    #print(svc_rbf_c1000.predict([[0.404695112, 0.556220642, 0.435032654, 0.95375712, 0.616367614, 0.140135251, 0, 0.801199482]]))
    #print(svc_rbf_c1000.predict([[0.572075354,	0.425227225,	0.196546918,	0.873134995,	0.922964603,	0.138429271,	0.159571425,	0.449790328]]))
    #print(svc_rbf_c1000.predict([[0.658484677,	0.443266785,	0.494248695,	0.944447685,	0.579502115,	0.503946736,	0.162076535,	0.779742237]]))
    
    # compute and print accuracy score
    print('Model accuracy score with rbf kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

    #13. Run SVM with linear kernel 

    #Run SVM with linear kernel and C=1.0
    # instantiate classifier with linear kernel and C=1.0
    linear_svc=SVC(kernel='linear', C=1.0) 


    # fit classifier to training set
    linear_svc.fit(X_train,y_train)


    # make predictions on test set
    y_pred_test=linear_svc.predict(X_test)
    
    tes=linear_svc.predict(X)
    data = []
    for i in tes:
        data.append([i])
    df_image_vide = pd.DataFrame(data, columns = ['target'])
    file_csv = pd.concat([df_image_vide])
    file_csv.to_csv(os.path.join(args.out,'data_pred_normaliz.csv'),index=False)
    print("save file csv")
    print(y_pred_test)

    #print("test pridect normaliz linear_svc")
    #print(linear_svc.predict([[0.5219771937229528, 0.37179858049219616, 0.4502084703084542, 0.9180453455852107, 0.21358339001686932, 0.0075136764207132415, 0.0, 0.522671169804032]]))
    #print(linear_svc.predict([[0.521977193677495, 0.371798580483986, 0.450208469694346, 0.918045345824172, 0.213583390236258, 0.00751367642071324, 0.0, 0.522671169302097]]))
    #print(linear_svc.predict([[0.404695112, 0.556220642, 0.435032654, 0.95375712, 0.616367614, 0.140135251, 0, 0.801199482]]))
    #print(linear_svc.predict([[0.572075354,	0.425227225,	0.196546918,	0.873134995,	0.922964603,	0.138429271,	0.159571425,	0.449790328]]))
    #print(linear_svc.predict([[0.658484677,	0.443266785,	0.494248695,	0.944447685,	0.579502115,	0.503946736,	0.162076535,	0.779742237]]))

    # compute and print accuracy score
    print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

    from sklearn.model_selection import cross_val_score
    from sklearn import svm

    # compute ROC AUC

    from sklearn.metrics import roc_auc_score

    #ROC_AUC = roc_auc_score(y_test, y_pred_test)

    #print('ROC AUC : {:.4f}'.format(ROC_AUC))

    from sklearn.model_selection import cross_val_score, cross_val_predict

    #Cross_validated = cross_val_score(linear_svc, X_train, y_train, cv=10).mean()
    #C_V = cross_val_predict(linear_svc, X_train, y_train, cv=10)

    #print(C_V)    

    #print('Cross validated : {:.4f}'.format(Cross_validated))

    #clf = svm.SVC(kernel='linear', C=1, random_state=42)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print('Model accuracy score with linear kernel and C=1.0 and cross_val_score : {}', scores)
    
    #Compare the train-set and test-set accuracy
    #Now, I will compare the train-set and test-set accuracy to check for overfitting.

    y_pred_train = svc_rbf_c1000.predict(X_train)

    print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

    #Check for overfitting and underfitting
    # print the scores on training and test set

    print('Training set score: {:.4f}'.format(svc_rbf_c1000.score(X_train, y_train)))

    print('Test set score: {:.4f}'.format(svc_rbf_c1000.score(X_test, y_test)))

    import pickle
    # save the model to disk
    filename = os.path.join(args.out, 'svc_rbf_c100_model_normaliz_2Class.sav')
    pickle.dump(svc_rbf_c100, open(filename, 'wb'))
    print('save model')
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test,y_test)
    print(result)

if __name__ == '__main__':

    main()