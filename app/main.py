from fastapi import FastAPI
from fastapi import FastAPI
#import schema
import uvicorn


#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import pandas as pd 

# Keras
import tensorflow
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


app = FastAPI()

def evaluate_model(y_test_class, y_pred_class):
    #cla_report = classification_report(y_test_class, y_pred_class)
    #confusion_mat = 0#pd.crosstab(y_test_class, y_pred_class, rownames=['Actual'], colnames=['Predicted'], margins=True)
    p_w,r_w,f_w,_ = precision_recall_fscore_support(y_test_class, y_pred_class, average='weighted')
    p_m,r_m,f_m,_ = precision_recall_fscore_support(y_test_class, y_pred_class, average='macro')
    return p_w,r_w,f_w, p_m,r_m,f_m

@app.get("/Cardiac_Disease_Classification/evaluation1/", tags=["Model 1 Evaluation"])
def create():

    x_test = pd.read_csv("./app/Testing_set_features.csv").to_numpy()
    y_test = pd.get_dummies(pd.read_csv("./app/Testing_set_Labels.csv")["Labels"]).to_numpy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_test = scaler.fit_transform(x_test)

    loaded_model = keras.models.load_model('./app/model_3.h5')
    y_pred = loaded_model.predict(x_test)
    y_pred_class = np.argmax(y_pred, axis=1)+1
    y_test_class = np.argmax(y_test, axis=1)+1
    p_w,r_w,f_w, p_m,r_m,f_m = evaluate_model(y_test_class, y_pred_class)
    return {"Precision_Weighted :":round(p_w,2), "Recall_Weighted ":round(r_w,2), "F1_Weighted ":round(f_w,2),"Precision_Macro :":round(p_m,2), "Recall_Macro ":round(r_m,2), "F1_Macro ":round(f_m,2)}

@app.get("/Cardiac_Disease_Classification/evaluation2/", tags=["Model 2 Evaluation"])
def create():

    x_test = pd.read_csv("./app/Testing_set_features.csv").to_numpy()
    y_test = pd.get_dummies(pd.read_csv("./app/Testing_set_Labels.csv")["Labels"]).to_numpy()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_test = scaler.fit_transform(x_test)

    loaded_model = keras.models.load_model('./app/model_3.h5')
    y_pred = loaded_model.predict(x_test)
    y_pred_class = np.argmax(y_pred, axis=1)+1
    y_test_class = np.argmax(y_test, axis=1)+1
    cla_report, confusion_mat = evaluate_model(y_test_class, y_pred_class)
    return {"Classification Report :":cla_report, "Confusion Matrix ":confusion_mat}

@app.post("/Cardiac_Disease/prediction/", tags=["Disease Prediction"])
def load_features():
    
    x_test = pd.read_csv("./app/Testing_set_features.csv").to_numpy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_test = scaler.fit_transform(x_test)

    loaded_model = keras.models.load_model('./app/model_3.h5')
    y_pred = loaded_model.predict(x_test[0:1])
    y_pred_class = int(np.argmax(y_pred, axis=1))+1
    
    #feature = features.dict()
    #feature = pd.DataFrame(feature, index=[0])
    #pred = loaded_model.predict(feature)
    map = {1:"Amyloidosis", 2:"Fabry",3:"HCM",4:"HTN",5:"Healthy"}
    disease = map[y_pred_class]
    return {"Predicted Label :":y_pred_class, "Predicted Disease :": disease}

#if __name__ == "__main__":
#    uvicorn.run("main:app", host="0.0.0.0", port=80)
