import joblib
import os
import numpy as np
import skimage
from skimage.io import imread , imshow
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import sys
from sklearn import svm

from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer

def resize_all(src, pklname, include, width=150, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary, 
    together with labels and metadata. The dictionary is written to a pickle file 
    named '{pklname}_{width}x{height}px.pkl'.

    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """

    height = height if height is not None else width

    data = dict()
    data['description'] = 'resized ({0}x{1})SIGN images in rgb'.format(
        int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    pklname = f"{pklname}_{width}px.pkl"

    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)

            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height)) #[:,:,::-1]
                    data['label'].append(subdir[:])
                    data['filename'].append(file)
                    data['data'].append(im)

        joblib.dump(data, './pkl_folder/'+pklname)


class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])

class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """

    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)

        try:  # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

if __name__ =='__main__':
    data_path = './Training_data/DST_DATA'
    base_name = 'Traffic-sign'
    width = 64 # ? It is the size of image after resized.
    include = {'STOP_LINE', 'STOP_THEN_GO', 'NR', 'NSTOP', 'OR','NONE','NL'} # * Identify seven kinds of traffic sign

    resize_all(src=data_path, pklname=base_name, width=width, include=include)
    data = joblib.load(f'./pkl_folder/{base_name}_{width}px.pkl')


    X = np.array(data['data'])
    y = np.array(data['label'])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )


    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(
        pixels_per_cell=(14, 14),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm='L2-Hys'
    )
    scalify = StandardScaler()


    X_train_gray = grayify.fit_transform(X_train)
    X_train_hog = hogify.fit_transform(X_train_gray)
    X_train_prepared = scalify.fit_transform(X_train_hog)


    svm_clf = svm.SVC()
    svm_clf.fit(X_train_prepared, y_train)

    X_test_gray = grayify.transform(X_test)
    X_test_hog = hogify.transform(X_test_gray)
    X_test_prepared = scalify.transform(X_test_hog)
    y_pred = svm_clf.predict(X_test_prepared)

    print(np.array(y_pred == y_test)[:25])
    print('')
    print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

    joblib.dump(svm_clf, f'./pkl_folder/Traffic-sign_model_{width}.pkl')
    joblib.dump(grayify, f'./pkl_folder/Traffic-sign_grayify_{width}.pkl')
    joblib.dump(hogify, f'./pkl_folder/Traffic-sign_hogify_{width}.pkl')
    joblib.dump(scalify, f'./pkl_folder/Traffic-sign_scalify_{width}.pkl')
