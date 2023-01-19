from fastapi import FastAPI
from fastapi import FastAPI
from fastapi import UploadFile, File
#import schema
import uvicorn

#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import pandas as pd 
import radiomics 
import six
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import SimpleITK as sitk
import os



# Keras
#import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
#import keras.backend as K
# Train-Test
import sklearn
from sklearn.model_selection import train_test_split
# Scaling data
from sklearn.preprocessing import StandardScaler
# Classification Report
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler


#Path
imagePath = "./app/Amyloid_x001A_0011960_20190211_im1.nii"
maskPath = "./app/Amyloid_x001A_0011960_20190211_im1mask.nii"
paramPath = "./app/params25.yaml"
keysPath = "./app/keys.npy"

x_test_path = "./app/Testing_set_features.csv"
y_test_path = "./app/Testing_set_Labels.csv"
model_path = './app/model_3.h5'

app = FastAPI()


def radiomics_data(imagePath, maskPath, paramPath):
    extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
    result = extractor.execute(imagePath, maskPath)
    features = {}
    for key, value in six.iteritems(result):
        features[key] = value
    keys = np.load(keysPath, allow_pickle=True)
    featured_cols = {}
    for k in keys:
        featured_cols[k]=features[k]
    featured_cols = pd.DataFrame(data = featured_cols, index=[0])
    return featured_cols.to_numpy()



def evaluate_model(y_test_class, y_pred_class):
    #cla_report = classification_report(y_test_class, y_pred_class)
    #confusion_mat = 0#pd.crosstab(y_test_class, y_pred_class, rownames=['Actual'], colnames=['Predicted'], margins=True)
    p_w,r_w,f_w,_ = precision_recall_fscore_support(y_test_class, y_pred_class, average='weighted')
    p_m,r_m,f_m,_ = precision_recall_fscore_support(y_test_class, y_pred_class, average='macro')
    return p_w,r_w,f_w, p_m,r_m,f_m

@app.get("/Cardiac_Disease_Classification/evaluation1/", tags=["Model 1 Evaluation"])
def create():

    x_test = pd.read_csv(x_test_path).to_numpy()
    y_test = pd.get_dummies(pd.read_csv(y_test_path)["Labels"]).to_numpy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_test = scaler.fit_transform(x_test)

    loaded_model = keras.models.load_model(model_path)
    y_pred = loaded_model.predict(x_test)
    y_pred_class = np.argmax(y_pred, axis=1)+1
    y_test_class = np.argmax(y_test, axis=1)+1
    p_w,r_w,f_w, p_m,r_m,f_m = evaluate_model(y_test_class, y_pred_class)
    return {"Precision_Weighted :":round(p_w,2), "Recall_Weighted ":round(r_w,2), "F1_Weighted ":round(f_w,2),"Precision_Macro :":round(p_m,2), "Recall_Macro ":round(r_m,2), "F1_Macro ":round(f_m,2)}

"""
@app.get("/Cardiac_Disease_Classification/evaluation2/", tags=["Model 2 Evaluation"])
def create():

    x_test = pd.read_csv(x_test_path).to_numpy()
    y_test = pd.get_dummies(pd.read_csv(y_test_path)["Labels"]).to_numpy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_test = scaler.fit_transform(x_test)

    loaded_model = keras.models.load_model(model_path)
    y_pred = loaded_model.predict(x_test)
    y_pred_class = np.argmax(y_pred, axis=1)+1
    y_test_class = np.argmax(y_test, axis=1)+1
    p_w,r_w,f_w, p_m,r_m,f_m = evaluate_model(y_test_class, y_pred_class)
    return {"Precision_Weighted :":round(p_w,2), "Recall_Weighted ":round(r_w,2), "F1_Weighted ":round(f_w,2),"Precision_Macro :":round(p_m,2), "Recall_Macro ":round(r_m,2), "F1_Macro ":round(f_m,2)}


@app.post("/Cardiac_Disease/prediction/", tags=["Disease Prediction"])
def load_features():
    
    #x_test = pd.read_csv("./app/Testing_set_features.csv").to_numpy()
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #x_test = scaler.fit_transform(x_test)

    loaded_model = keras.models.load_model(model_path)
    data_point = radiomics_data(imagePath, maskPath,paramPath)

    y_pred = loaded_model.predict(data_point)
    y_pred_class = int(np.argmax(y_pred, axis=1))+1
    
    map = {1:"Amyloidosis", 2:"Fabry",3:"HCM",4:"HTN",5:"Healthy"}
    disease = map[y_pred_class]
    return {"Predicted Label :":y_pred_class, "Predicted Disease :": disease}

"""
@app.post("/Cardiac_Disease/prediction1/", tags=["Disease Prediction"])
async def load_predict(Image: UploadFile = File(...), Mask: UploadFile = File(...)):
    
    file_location = "./app/"
    imagename = Image.filename
    maskname = Mask.filename
    
    image_content = await Image.read()
    mask_content = await Mask.read()
 
    image_path = file_location+imagename
    mask_path = file_location+maskname

    with open(image_path, "wb") as i_f:
        i_f.write(image_content)

    with open(mask_path, "wb") as m_f:
        m_f.write(mask_content)

    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    
    loaded_model = keras.models.load_model(model_path)
    data_point = radiomics_data(image, mask, paramPath)


    y_pred = loaded_model.predict(data_point)
    y_pred_class = int(np.argmax(y_pred, axis=1))+1
    
    map = {1:"Amyloidosis", 2:"Fabry",3:"HCM",4:"HTN",5:"Healthy"}
    disease = map[y_pred_class]

    os.remove(image_path)
    os.remove(mask_path)

    return {"Predicted Label :":y_pred_class, "Predicted Disease :": disease}

"""
@app.post("/upload-file/")
async def create_upload_file(file: UploadFile = File(...)):    
    file_location = "./"
    filename = file.filename
    file_content = await file.read()
    generated_name = file_location+filename

    with open(generated_name, "wb") as f:
        f.write(file_content)
    return{"done":"done"}

"""    

#if __name__ == "__main__":
#    uvicorn.run("main:app", host="0.0.0.0", port=80)
