import os
from posix import listdir
from shutil import copyfile

#from app import app
from flask import Flask, render_template, request, redirect
from flask.wrappers import Request
from create_annotation import create_annotation
from convert_data import main
from res_final import main_res

skills_app = Flask(__name__)

#app.config["IMAGE_UPLOADS"] = "E:/2 MASTER/Memoire/06-28-2021/U-net-Lung-Segmentation-main/U-net-Lung-Segmentation-main/static/css/img/uploads"
IMAGE_UPLOADS = "E:/2 MASTER/Memoire/06-28-2021/U-net-Lung-Segmentation-main/U-net-Lung-Segmentation-main/static/img/uploads"

@skills_app.route("/", methods=["GET", "POST"])
def homepage():

    path_img = "static/img"
    load_lung_model = "model/best_checkpoint[epoch_49].pt"
    loader_model_svm = "model/linear_svc_model_normaliz_version02.sav"
    data_not_normaliz = "model/data_concat_non_normaliz.csv"

    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image_name = image.filename
            list1 = image_name.split('.')

            for i in list1:
                if (i == 'png' or i == 'jpeg' or i == 'jpg'):
                    format_img = i
            image.save(os.path.join(IMAGE_UPLOADS, 'image_test.'+format_img))

            # create path image operation
            if os.path.exists(path_img+"/image_operation"):
                os.rmdir(path_img+"/image_operation")
            if not os.path.exists(path_img+"/image_operation"):
                os.mkdir(path_img+"/image_operation")
            
            # copy image test to image operation
            copyfile(os.path.join(path_img,'uploads','image_test.'+format_img),
                os.path.join(path_img,'image_operation','image_test.'+format_img))

            # copy image help to image operation
            img_h_list = listdir(path_img+"/image_help")
            for n_img in img_h_list:
                copyfile(os.path.join(path_img,'image_help',os.path.basename(n_img)),
                    os.path.join(path_img,'image_operation',os.path.basename(n_img)))

            # create path out convert data
            if os.path.exists(path_img+"/convert_data"):
                os.rmdir(path_img+"/convert_data")
            if not os.path.exists(path_img+"/convert_data"):
                os.mkdir(path_img+"/convert_data")
            
            # convert data 
            main(path_img+"/convert_data", load_lung_model, path_img, 'image_operation')

            # create path out result svm
            if os.path.exists(path_img+"/file_csv"):
                os.rmdir(path_img+"/file_csv")
            if not os.path.exists(path_img+"/file_csv"):
                os.mkdir(path_img+"/file_csv")
            
            # result svm
            result = main_res(path_img+"/file_csv", data_not_normaliz, path_img+"/convert_data", loader_model_svm)

            print("result = ",result)

            print("image saved")
            
            print(image_name)
            print(request.url)
            return redirect(request.url, res = result)
        
    return render_template("upload.html")

if __name__ == "__main__":
    skills_app.run(port=9000)