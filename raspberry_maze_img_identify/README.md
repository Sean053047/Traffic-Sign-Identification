# **Module description**
>Here, I will illustrate three modules of project about features and how they work.   

Following are three modules I will talk about :
* _Image_processing.py_
* _Sign_detect.py_
* _Training.py_
## **_Image_processing.py_**
>_Image_processing.py_ aims to deal with raw data to capture traffic sign out from raw image.  

**Usage**
> img = cv2.imread(_"image_path"_)  
mask, color = get_sign(img)  
align = align_img(img, mask, color)

**Variable description**
|Variable | description |
|:--:|:--|
|_img_|Raw data of image|
|_mask_|Binary image with effective region of sign|
|_color_|Main color of sign|
|_align_|{_size_}*{_size_} pxl resized image of traffic sign|

**Function description**  
#### **_get_sign_** 
There are three main kinds of colors of traffic sign which would appear in this project :
1. red 
2. blue 
3. white  

>Step 1:
>It will use the difference of b&g,b&r  and difference of r&b, r&g to get the blue region and red region individually. _"mask"_ is a binary image which represents the effective region of sign.  

>Step 2:  
Compare the magnitude of blue regioin with red region to determinate that raw image is blue sign or red sign. If both area of red mask and blue mask are not large enough, raw image will be treated as white sign and will be processed by _get_white_ function.  

>Final Step:  
After Comparison, we can gain the effective region  of traffic sign (_mask_) and the name of main color (_color_). These variables will be returned from this function.
#### **_get_white_** 
>_get_white_ function uses mean and standard deviation of image to remove the background of raw image so as to pretain the portion of white stop line. If there isn't any sign in image, you alse can send it into the functoin. The result can be predict to be _"NONE"_ which make the car do itself as normal. 

#### **_align_img_**

>Step 1:  
Based on _"color"_, function chooses different kernel to process the _"mask"_ with morphology.

>Step 2:  
Use _"mask"_ to get the contour with maximum area. 
After that, using _cv2.minAreaRect()_ and _cv2.boxPoints()_ can gain the four end points of rotated rectangle enclosing contour with minimum area. That four points is what we want. 

>Final Step:  
Use the four end points to get the transform matrix of perspective transform to resize the image. Then you can get the {_size_}*{_size_} aligned image which has a good representative of traffic sign. 
## **_Training.py_**
>Before training, you should check the representative of training data. Discard the unwanted aligned image. This operation can rise your accuracy.  
Data which were processed by _Image_processing.py_ will later be processed by  
>1. _Resize_ ( {_size_}*{_size_} img ), 
>2. _RGB2Gray_, 
>3. _HogTransform_ , 
>4. _StandardScaler_.     

>After that, I use **_svm.SVC()_** as ML model to fit prepared training data. 

>Finally, we can get the model which reach **98.7%** accuracy. After training, this module dumps four models which would be used when you want to gain a predictoin. Four models are listed below :
>* _Traffic-sign_grayify_64.pkl_
>* _Traffic-sign_hogify_64.pkl_
>* _Traffic-sign_scalify_64.pkl_
>* _Traffic-sign_model_64.pkl_
## **_Sign_detect.py_**
>Before prediction, this module should import three classes such as "_RGB2GrayTransformer_", "_HogTransformer_" from "_Training_", "_StandardScaler_"  from "_sklearn.preprocessing_" and "_svm_" from "_sklearn_". Then the module knows what loaded pickle files are and it can operate succesfully.
Raw image follows the same steps as training before fitting. Then it will be send into model to predict which sign it is.

**Usage**
>img = cv2.imread("image_path")  
>predict , align = predict_from_single(img)  

**Variable description**
|Variable | description |
|:--:|:--|
|_img_|Raw data of image|
|_predict_|Prediction of unknow image|
|_align_|{_size_}*{_size_} pxl resized image of traffic sign|
