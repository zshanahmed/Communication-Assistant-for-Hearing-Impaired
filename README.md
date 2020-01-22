# Communication Assistant for Hearing Impaired
A bi-directional communication assistant designed to facilitate communication between a hearing impaired person and a normal person. The system involves conversion of hand gestures used by hearing impaired persons into human readable text for a normal person to read. We have used SURF Feature Detector for collection of useful features in our dataset. Moreover we have used Bag-of-Words to build a descriptive dictionary based on those distinguishing features. To display sign language images from text entered by a normal person, we used Hash Tables. To have an overview about the project you can watch the video [here](https://www.youtube.com/watch?v=DqZKGPqf_6U).

## Introduction
### Gesture to Text
#### Data Collection and Pre-processing
In our scenario we collected data for 10 gestures. Moreover, we collected 100 samples per gesture. To have accurate data pertaining to Pakistan Sign Language (Hand gestures used in Pakistan), we visited a local school dedicated especially for hearing impaired people. 

After extensive collection of data, we fine-tuned the threshold to find skin-colored contours from the image. Images were originally taken in RGB scale. But to have an luminance independent scenario, we used *YCrCb* to separate the luminance (intensity) information from the color information. After finding the threshold we fine-tuned the data for pruning of any unwanted samples

#### Feature Detection, Extraction and Matching
For feature detection we used binary images. As far as feature descriptors are concerned, we used **Speeded Up Robust Features (SURF)**. As *SURF* uses integral images instead of Difference of Gaussians (DoG) to approximate Laplacian of Gaussian (LoC), it is quite faster then other major descriptors such as **Scale Invariant Feature Transform (SIFT)**. Moreover SURF is orientation-independent. After extraction of features from *SURF* we used **Bag of Words (BoG)** to cluster similar features together. 

This is how SURF descriptors look like for the gesture of Day:
 <p align="center">
  <img width="460" height="300" src="https://i.imgur.com/eK3FcqB.png">
</p>

#### Classification
For classification we used **Support Vector Machines (SVM)**. Specifically we used C-Support Vector Classification (CSVC) to penalize any outliers that come across during classification. We used **Radial Basis Function (RBF)** kernel for proper classification by SVM based on our data. 

#### Testing Phase
To calculate accuracy we ran our system over 15 images per gesture. The accuracy turned out to be *88%* for the test data set, and *86%* when those gestures were done in real-time

### Text to Gesture
#### Text Filtering
Only the important words conveying the context of the sentence were filtered out and any conjunction and helping verbs were removed from the sentence

#### Text Matching
The filtered text was then passed into our database and for matched words the related image of hand gesture will be displayed to the hearing-impaired person.

#### Communication
To have communication on different devices between the hearing impaired person and the normal person we followed a client-server protocol to establish the corresponding system. 

## Usage 
1. Configure your favorite IDE with OpenCv. We used Visual studio and the information about its configuration can be found [here](https://docs.opencv.org/2.4/doc/tutorials/introduction/windows_visual_studio_Opencv/windows_visual_studio_Opencv.html). For our project we used OpenCv 2.4.10
2. After successfully setting up path of open-cv libraries in your project. Compile and run the code named 'Hearing_Impaired.cpp'. We are assuming that Hearing Impaired Person is working as Server in this Client-Server Architecture
3. The server will be waiting for incoming connections. Compile and Run the code named 'Normal_Person.cpp'
4. **Be sure that you have classification.xml and dictionary.yml files in your project folder**
