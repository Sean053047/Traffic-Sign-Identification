# Traffic Sign Identification

## **Introduction**
This project aims to use smartcar with Raspberry to go through a maze.  
During the process, smartcar must do certain motivation when it identifies corresponding traffic sign.   
Following images are the results of during process. Take traffic sign - "Turn right only" for example.
1. Assume This image is the picture captured by camera.

!(Raw image)[https://drive.google.com/file/d/1KUACezDVrgfG54vF8nw-QEAj5_8Mbzh5/view]

2. I can get the mask of traffic sign by means of image processing. 

!(Raw image)[https://drive.google.com/file/d/1aGDd0UG4ng6w91vbMs7ZHc5GyndRJ_ie/view]

3. Resize the effective region to make sign significant and also be higher representative than raw image. After that, we can send it into model to predict what kind of sign it is.

!(Raw image)[https://drive.google.com/file/d/1lBO8yZUJzd66qUizP8j9uVS3qIP9xu2q/view]

Here is our team's [_demo video_](https://youtube.com/shorts/5QEpH4niNis?feature=share) of competition.  
In the maze, There are seven kinds of sign which should be recognized listed below :
|Traffic Sign|Description|
|:--:|:---|
|NL|Do not turn left.|
|NONE| None of sign in image.|
|NR|Do not turn right.|
|NSTOP|Do not stop the car.|
|OR|Turn right only.|
|STOP_LINE|Stop car first, then go forward. (White line)| 
|STOP_THEN_GO|Stop car first, then go forward. (Red octagonal sign)|

Besides project codes, I also push some raw data onto the repository. If you are interesting in this, you can download data and do it by yourself.

Following section, I will give you a brief introduction to let you know how those function work.

## **Requirement**
### Python: 3.10
| Package| Versions|
|:---:|:---:|
|Numpy|1.24.1|
|opencv-contrib-python|4.7.0.68|
|joblib|1.2.0|
|scikit-learn|1.2.0|
|scikit-image|0.19.3|
