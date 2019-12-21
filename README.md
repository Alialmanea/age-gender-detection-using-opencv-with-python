# face_detection_using_opencv_with_python


<h3>Introduction</h3>
Age and gender, two of the key facial attributes, play a very foundational role in social interactions, making age and gender estimation from a single face image an important task in intelligent applications, such as access control, human-computer interaction, law enforcement, marketing intelligence
and visual surveillance, etc.

<h5>Real world use-case :</h5>

Recently I came across Quividi which is an AI software application which is used to detect age and gender of users who passes by based on online face analyses and automatically starts playing advertisements based on the targeted audience.
Another example could be AgeBot which is an Android App that determines your age from your photos using facial recognition. It can guess your age and gender along with that can also find multiple faces in a picture and estimate the age for each face.
Inspired by the above use cases we are going to build a simple Age and Gender detection model in this detailed article. So let's start with our use-case:
Use-case — we will be doing some face recognition, face detection stuff and furthermore, we will be using CNN (Convolutional Neural Networks) for age and gender predictions from a youtube video, you don’t need to download the video just the video URL is fine. The interesting part will be the usage of CNN for age and gender predictions on video URLs.

<h6>Source:https://www.kiwi-digital.com/produkty/age-gender-detection</h6>


<h5>Requirements :</h5>

1.pip install OpenCV-python</br> 
2.Haar cascades for Face detection</br>  
3.Gender Recognition with CNN</br>  
4.Age Recognition with CNN</br> 

 <h6>For Download</h6> 
 opencv link it here:https://opencv.org
 Hear cascade link it here:https://github.com/opencv/opencv/blob/master/data/haarcascades
 prtotxt and .caffemodel from this link :https://talhassner.github.io/home/publication/2015_CVPR

How it works

Let's have an overview how it works in general.


<a href="https://imgflip.com/gif/3k4yh4"><img src="https://i.imgflip.com/3k4yh4.gif" title="made at imgflip.com"/></a>


First, the photo is taken from the webcam stream live by the cv2 module.
