# age & gender detection-using-opencv-with-python




<a href="https://imgflip.com/gif/3k54ya"><img src="https://i.imgflip.com/3k54ya.gif" title="made at imgflip.com"/></a>



<h3>Introduction</h3>
Age and gender, two of the key facial attributes, play a very foundational role in social interactions, making age and gender estimation from a single face image an important task in intelligent applications, such as access control, human-computer interaction, law enforcement, marketing intelligence
and visual surveillance, etc.

<h5>Real world use-case :</h5>

Recently I came across Quividi which is an AI software application which is used to detect age and gender of users who passes by based on online face analyses and automatically starts playing advertisements based on the targeted audience.
Another example could be AgeBot which is an Android App that determines your age from your photos using facial recognition. It can guess your age and gender along with that can also find multiple faces in a picture and estimate the age for each face.
Inspired by the above use cases we are going to build a simple Age and Gender detection model in this detailed article. So let's start with our use-case:
Use-case — we will be doing some face recognition, face detection stuff and furthermore, we will be using CNN (Convolutional Neural Networks) for age and gender predictions from a youtube video, you don’t need to download the video just the video URL is fine. The interesting part will be the usage of CNN for age and gender predictions on video URLs.

<h6>Source:https://www.kiwi-digital.com/produkty/age-gender-detection</h6>


<h3>What is Computer Vision?</h3>
Computer Vision is the field of study that enables computers to see and identify digital images and videos as a human would. The challenges it faces largely follow from the limited understanding of biological vision. Computer Vision involves acquiring, processing, analyzing, and understanding digital images to extract high-dimensional data from the real world in order to generate symbolic or numerical information which can then be used to make decisions. The process often includes practices like object recognition, video tracking, motion estimation, and image restoration.

<h3>What is OpenCV?</h3>
OpenCV is short for Open Source Computer Vision. Intuitively by the name, it is an open-source Computer Vision and Machine Learning library. This library is capable of processing real-time image and video while also boasting analytical capabilities. It supports the Deep Learning frameworks TensorFlow, Caffe, and PyTorch.

<h3>What is a CNN?</h3>
A Convolutional Neural Network is a deep neural network (DNN) widely used for the purposes of image recognition and processing and NLP. Also known as a ConvNet, a CNN has input and output layers, and multiple hidden layers, many of which are convolutional. In a way, CNNs are regularized multilayer perceptrons.

<h5>Gender and Age Detection Python Project- Objective</h5>

To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture using Deep Learning on the Adience dataset.

<h5>Gender and Age Detection – About the Project</h5>

In this Python Project, we will use Deep Learning to accurately identify the gender and age of a person from a single image of a face. We will use the models trained by Tal Hassner and Gil Levi. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, we make this a classification problem instead of making it one of regression.

The CNN Architecture
The convolutional neural network for this python project has 3 convolutional layers:

Convolutional layer; 96 nodes, kernel size 7
Convolutional layer; 256 nodes, kernel size 5
Convolutional layer; 384 nodes, kernel size 3

 
It has 2 fully connected layers, each with 512 nodes, and a final output layer of softmax type.

To go about the python project, we’ll:

Detect faces
Classify into Male/Female
Classify into one of the 8 age ranges
Put the results on the image and display it
The Dataset
For this python project, we’ll use the Adience dataset; the dataset is available in the public domain and you can find it here. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models we will use have been trained on this dataset.

<h5>Requirements :</h5>

1.pip install OpenCV-python</br>2.Haar cascades for Face detection</br>3.Gender Recognition with CNN</br>4.Age Recognition with CNN</br> 

 <h6>For Download</h6> 
 opencv link it here:https://opencv.org</br> 
Hear cascade link it here:https://github.com/opencv/opencv/blob/master/data/haarcascades
 prtotxt and .caffemodel from this link :https://talhassner.github.io/home/publication/2015_CVPR

How it works

Let's have an overview how it works in general.


<a href="https://imgflip.com/gif/3k4yh4"><img src="https://i.imgflip.com/3k4yh4.gif" title="made at imgflip.com"/></a>


For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.

We use the argparse library to create an argument parser so we can get the image argument from the command prompt. We make it parse the argument holding the path to the image to classify gender and age for.
 
For face, age, and gender, initialize protocol buffer and model.

Initialize the mean values for the model and the lists of age ranges and genders to classify from.

<a href="https://imgflip.com/gif/3k65dx"><img src="https://i.imgflip.com/3k65dx.gif" title="made at imgflip.com"/></a>
