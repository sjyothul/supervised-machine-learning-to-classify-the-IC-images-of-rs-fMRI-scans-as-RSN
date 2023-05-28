import csv
import cv2
import glob
import os
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from PIL import Image
from sklearn import metrics

def templateMatch(file, template):
    org_img = cv2.imread(file)
    norm_img = np.zeros((800, 800))
    norm_temp = cv2.normalize(org_img, norm_img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.cvtColor(norm_temp, cv2.COLOR_BGR2GRAY)
    w,h = template.shape[::-1]    
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    thresh_val = 0.95
    ver = np.where(result >= thresh_val)[::-1]
    return w,h,ver,org_img,img
        
def CV2ToPillow(img_cv2):
    img_arr = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pillow = Image.fromarray(img_arr)
    return img_pillow

def PillowToCV2(img_pillow):
    img_pillow = img_pillow.convert('RGB')
    img_arr = np.array(img_pillow)
    img_cv2 = img_arr[:,:,::-1]
    return img_cv2

path = os.getcwd()
target = str(path)+'/testPatient/'
path_dataset = glob.glob(str(path)+'/testPatient/test_Data/*_thresh.png')
template = cv2.imread(str(path)+'/template.png', 0)

temp = pd.DataFrame()
predict = pd.DataFrame(columns = ['IC_Number','Label'])
for dirname, _, filenames in os.walk(target):
    for filename in filenames:
        if(re.search("\_Labels.csv$", filename)):
            im = []
            csv_path = os.path.join(dirname, filename)
            a = pd.read_csv(csv_path)
            predict['IC_Number'] = a['IC']
            if temp.empty:
                temp['IC'] = a['IC']
                temp['Label'] = a['Label'].apply(lambda x : 1 if(x > 0) else 0)
                for k in range(1,len(a)+1):
                    loc = str(dirname)+'test_Data/IC_'+str(k)+'_thresh.png'
                    wid,hei,dim,img_org,img_gray = templateMatch(loc, template)
                    for vtx in zip(*dim):
                        img = CV2ToPillow(img_org)
                        img_slice = img.crop((vtx[0], vtx[1], vtx[0] + 944, vtx[1] + 708))
                        img_req = PillowToCV2(img_slice)
                        break
                    img_100x100 = cv2.resize(img_req, (224, 224))
                    img_rgb = cv2.cvtColor(img_100x100, cv2.COLOR_BGR2RGB)
                    img_arr = img_rgb.flatten()
                    im.append(img_arr)
                temp['Image'] = im
            else:
                tempDf = pd.DataFrame(columns = ['IC','Label','Image'])
                tempDf['IC'] = a['IC']
                tempDf['Label'] = a['Label'].apply(lambda x : 1 if(x > 0) else 0)
                for k in range(1,len(a)+1):
                    loc = str(dirname)+'test_Data/IC_'+str(k)+'_thresh.png'
                    wid,hei,dim,img_org,img_gray = templateMatch(loc, template)
                    for vtx in zip(*dim):
                        img = CV2ToPillow(img_org)
                        img_slice = img.crop((vtx[0], vtx[1], vtx[0] + 944, vtx[1] + 708))
                        img_req = PillowToCV2(img_slice)
                        break
                    img_100x100 = cv2.resize(img_req, (224, 224))
                    img_rgb = cv2.cvtColor(img_100x100, cv2.COLOR_BGR2RGB)
                    img_arr = img_rgb.flatten()
                    im.append(img_arr)
                tempDf['Image'] = im
                temp = pd.concat([temp, tempDf])
print(temp)
n_classes = temp['Label'].unique()
num_class = temp['Label'].nunique()

img_lst = []
for arr in temp['Image']:
    img_lst.append(arr)
X = np.array(img_lst)
X = X.astype('float32')
X = X/255.
Y = temp['Label']

print(Y)

img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 1)
X_test = X.reshape(X.shape[0], img_rows, img_cols, 3)
Y_test_one_hot = tf.keras.utils.to_categorical(Y, num_classes = num_class)

save_at = str(path)+"/trained_model.hdf5"
model_best = tf.keras.models.load_model(save_at)
score = model_best.evaluate(X_test, Y_test_one_hot, verbose = 1)
print(score)
print('Accuracy over the test set: \n ', round((score[1]*100), 2), '%')

Y_pred = np.round(model_best.predict(X_test))

pred_score = model_best.evaluate(X_test, Y_pred, verbose = 1)
print(pred_score)
print('Accuracy over the pred set: \n ', round((pred_score[1]*100), 2), '%')

label = []
for i in range(len(Y_pred)):
    if(Y_pred[i][0] == 1.0):
        label.append(0)
    elif(Y_pred[i][1] == 1.0):
        label.append(1)
    elif(Y_pred[i][0] == 0.0):
        continue
    else:
        continue
predict['Label'] = label
loc_predict = str(path)+'/Results.csv'
predict.to_csv(loc_predict, index = False)

confusion = metrics.confusion_matrix(list(Y), label)
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
total = (TP+TN+FP+FN)

accuracy = (TP + TN)/total
misclassification_error = (FP + FN)/total
precision = TP/(TP + FP)
recall = TP/(TP + FN)
specificity = TN/(TN + FP)
sensitivity = TP/(TP + FN)
f1_score = (2 * TP)/((2 * TP) + FP + FN)

data = {
        'Accuracy':"{:.0%}".format(accuracy),
        'Misclassification-Error':"{:.0%}".format(misclassification_error),
        'Precision':"{:.0%}".format(precision),
        'Recall':"{:.0%}".format(recall),
        'F1-Score':"{:.0%}".format(f1_score),
        'Specificity':"{:.0%}".format(specificity),
        'Sensitivity':"{:.0%}".format(sensitivity)
    }
loc_data = str(path)+'/Metrics.csv'
with open(loc_data, 'w', newline = '') as f:
    writer = csv.writer(f)
    for row in data.items():
        writer.writerow(row)