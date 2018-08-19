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

<<<<<<< HEAD
At the beginning i was following the lecture and build a model with three layers of convolutional neural network and three layers of fully connected neural network. I played with different filter sizes and depth, as well as activation function, but it seems the car cannot drive properly. AS i noticed that validation loss is much higher than training loss, i added some dropout and l2 regularzation, but still it doesn't work well. After going through internet, i find NVIDIA architecture and tried with it.The car can drive one lap successfully.

NOTE: As the reviewer recommends nonlinear activation function and dropout, i further added 'tanh' activation function and dropout layers.
=======
At the beginning i was following the lecture and build a model with three layers of convolutional neural network and three layers of fully connected neural network. I played with different filter sizes and depth, as well as activation function, but it seems the car cannot drive properly. AS i noticed that validation loss is much higher than training loss, i added some dropout and l2 regularzation, but still it doesn't work well. After going through internet, i find NVIDIA architecture and tried with it. The car can drive one lap successfully.

>>>>>>> 83fb799deabb9b7d5d65c89b478272fb164fa3cb


<<<<<<< HEAD
#### 2. Attempts to reduce overfitting in the model

=======
>>>>>>> 83fb799deabb9b7d5d65c89b478272fb164fa3cb
Tried with dropout layers and l2 regularzer but didn't work so well. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

#### 4. Appropriate training data

<<<<<<< HEAD
As I was not able to drive in a proper way in the simulator using my laptop, I executed this project with the dataset provided by Udacity. I am aware that the smoothness of the car staying in the middle of the road and starting to turn when a corner arrives can be improved to add more data to my dataset. I would collect data from the following driving manoeuvres to improve my model:
- Drive from the side of the road to the middle. This at different parts of the track.
- Collect more driving data around parts of the track which are different compared to the biggest chunk of the track. For example, where there is a sand road exit, or where there is a transition from shade to sun on the road, etc.
- Around the bridge as the texture/color of the bridge compared to the rest of the track is completely different.
- I would drive the track backwards to make the model generalize better. However, this was virtually covered by flipping the original dataset.

As X_train/y_train dataset preprocessing I undertook the following:
- convert the images from BGR to RGB as the simulator will fed the trained model with RGB images. 
- loaded images from 3 camera's to have a better result in corners and pulling the car back to the middle of the road. For the left and right image, a correction factor was included to the steering angle to improve the change in direction when going off the middle of the road.
- for the complete dataset, images were then flipped and added to the dataset. This gave us more data to train on and data that now virtually was edited and doing a clockwize tour around the track instead of a anti-clockwize tour.
- during the training, using the Keras library, the crop function was used to cut off trees/sky at the top and the front of the car at the bottom. A positive side effect of this was that the number of pixels went down which speed up the training.
=======
Used provided training data. I noticed that the provided data has very few image on turning left or right or going through the bridge. That could be the reason my original architunderecture didn't work. I could manually added some more images especially for these cases.
>>>>>>> 83fb799deabb9b7d5d65c89b478272fb164fa3cb

### Model Architecture and Training Strategy

#### 1. Solution Design Approach



My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it can recognize traffic signs successfully.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it is not overfited.

But it seems the car was not driving well with this model. Then i tried with NVIDIA architecture and it turns out the car drives perfectly.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

<<<<<<< HEAD
The final model architecture:
There are 5 CNNs with a Relu for nonlinearity. For the first 3, 5x5 filters were used, for the last 2 3x3. The depths are as follows: 24, 36, 48, 64 and 64. After that the model is flattened and 4 FC layers are implemented: 100, 50, 10 and 1. The complete model is build up using Keras. 
(NVIDIA Architecture)

NOTE: As the reviewer recommends nonlinear activation function and dropout, i further added 'tanh' activation function and dropout layers in between of fully connected layers.
=======
The final model architecture  consisted of a convolution neural network with the following layers and layer sizes ...

>>>>>>> 83fb799deabb9b7d5d65c89b478272fb164fa3cb


#### 3. Creation of the Training Set & Training Process

Used provided training data.
NOt only centre images but also images from left and right cameras are used. Corresponding correction factors are used.
Note: all the images are read in in RGB.
To augment the data set, I also flipped images and angles thinking that this would increase the training dataset and may generalize the model better. 
After the collection process, I had 38572/0.8 number of data points. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by trail and error. I used an adam optimizer so that manually training the learning rate wasn't necessary.
