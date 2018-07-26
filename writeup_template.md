# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

At the beginning i was following the lecture and build a model with three layers of convolutional neural network and three layers of fully connected neural network. I played with different filter sizes and depth, as well as activation function, but it seems the car cannot drive properly. AS i noticed that validation loss is much higher than training loss, i added some dropout and l2 regularzation, but still it doesn't work well. After going through internet, i find NVIDIA architecture and tried with it. The car can drive one lap successfully.


#### 2. Attempts to reduce overfitting in the model

Tried with dropout layers and l2 regularzer but didn't work so well. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

#### 4. Appropriate training data

Used provided training data. I noticed that the provided data has very few image on turning left or right or going through the bridge. That could be the reason my original architunderecture didn't work. I could manually added some more images especially for these cases.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach



My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it can recognize traffic signs successfully.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it is not overfited.

But it seems the car was not driving well with this model. Then i tried with NVIDIA architecture and it turns out the car drives perfectly.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture  consisted of a convolution neural network with the following layers and layer sizes ...



#### 3. Creation of the Training Set & Training Process

Used provided training data.
NOt only centre images but also images from left and right cameras are used. Corresponding correction factors are used.
Note: all the images are read in in RGB.
To augment the data set, I also flipped images and angles thinking that this would increase the training dataset and may generalize the model better. 
After the collection process, I had 38572/0.8 number of data points. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by trail and error. I used an adam optimizer so that manually training the learning rate wasn't necessary.
