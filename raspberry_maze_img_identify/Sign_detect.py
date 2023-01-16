import joblib
import numpy as np
import skimage
from skimage.transform import resize
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
import cv2
import os
from Image_processing import align_img, get_sign
from time import sleep
from Training import RGB2GrayTransformer ,HogTransformer
from sklearn.preprocessing import StandardScaler
from sklearn import svm



model = joblib.load(
    './pkl_folder/Traffic-sign_model_64.pkl')
grayify = joblib.load(
    './pkl_folder/Traffic-sign_grayify_64.pkl')
hogify = joblib.load(
    './pkl_folder/Traffic-sign_hogify_64.pkl')
scalify = joblib.load(
    './pkl_folder/Traffic-sign_scalify_64.pkl')


def predict_from_single(img , size = 64):
    '''此 img 是由cv2 讀取近來的影像，預設為BGR，get_sign 跟 align 處理得 圖像'''
    mask, color = get_sign(img)
    align = align_img(img, mask, color)

    test = cv2.cvtColor(align, cv2.COLOR_BGR2RGB)
    test = resize(test, (size, size))  # skimage 的resize [:,:,::-1]

    gray = grayify.transform([test])
    hog = hogify.transform(gray)
    prepared = scalify.transform(hog)
    predict = model.predict(prepared)
    return *predict, align 

def predict_from_folder(src, size =64):
    
    for f in os.listdir(src):
        img = cv2.imread( os.path.join(src,f))
        mask, color = get_sign(img)
        align = align_img(img, mask, color)
        test = cv2.cvtColor(align, cv2.COLOR_BGR2RGB)
        
        test = resize(test, (size, size))  # [:,:,::-1]

        
        test_gray = grayify.transform([test])
        test_hog = hogify.transform(test_gray)
        test_prepared = scalify.transform(test_hog)

        predict = model.predict(test_prepared)
        yield *predict, align


if __name__ == '__main__':
    test_folder = './Test_data'
    prediction = predict_from_folder(test_folder , 64)

    for predict, img in prediction:
        cv2.imshow(f'{predict}', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

