#**Udacity Self-Driving Car Nanodegree - Project 3 - Behavioral Cloning** 

---

**Introduction**

The goal of this project was to apply transfer learning and deep learning techniques to autonomously drive a car in Udacity's simulated env.


[//]: # (Image References)

[image1]: ./nvidia_model.png "Model Visualization"
[image2]: ./center.jpg "Center Image"
[image3]: ./center_corr1.jpg "Center Correction Image"
[image4]: ./center_corr2.jpg "Center Correction Image"
[image5]: ./center_corr3.jpg "Center Correction Image"

## Project Notes

###Model Architecture and Training Strategy

####1. Architecture Approach

Initially, I started of with the LeNet model to quickly build out the pipeline to test the end to end flow of the program with Udacity's training set. This model quickly failed in the initial few seconds of testing the model in autonomous mode. Later, as mentioned in the project instructions, and as suggested in the forums, I adapted the NVDIA model as the basis for this project. The diagram for the model is shown below:

![alt text][image1]

The model starts out by applying Image normalization and Cropping using Keras Lambda and Cropping2D functions respectively (model.py line 82 and 85). Next it consists of three 5 x 5 convolution layers, followed by two 3x3 convolutional layers, and flattening layer followed by 3 fully-connected layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (20% split). I found that my first model had a realatively low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by introducing MaxPooling layers or strides 1x1 for the initial 3 convolutional layers. This improved the model's performance marginally. I then introduced Dropout layers with a prob factor of 0.5 after every fully connected layer to reduce overfitting. I also increased the validation split to 25%. This was the final design of my model. Another option to reduce overfitting was to increase the the training data. 

####2. Data Collection

After testing the model with just the training data that was supplied by Udacity and failing in a few spots, I began collecting my own training data for testing the model. This involved driving 2 laps around the track, 1 lap around the track in the opposite direction, 1 lap of corrective driving, where I veer close to the edges of the track and record the car being steered back to the center of the lane.

There were a few spots where the vehicle fell off the track (After crossing the Bridge). To improve the driving behavior in these cases, I appended more training data that involved taking corrective driving measures in such areas. At this point I also chose to use the left and right angle images along with the center channel.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. However, I was still not satisfied with the final MSE and Validation loss. The validation loss was still much higher compared to the MSE. Perhaps increasing the number of EPOCHs, or increasing the training data could yeild better. The total number of images used (Including center, left, and right) was just  16167. 

Here is an example image of center lane driving:

![alt text][image2]

These images show what a recovery looks like :

![alt text][image3]
![alt text][image4]
![alt text][image5]

####3. Future Optimizations  

I did try to augment the data set by flipping the images and angles to increase the size of the data set. However, this resulted in my model failing fairly quickly. The car never steered and kept continuing on a straight path until it veered off the track. I'm yet resolve this issue. I'm not sure as to why my model fails when introducting flipped data. The code for flipping the data is located in model.py beginning at line 48.

I also need to work on introducing a generator function to address increase in the training data (However, this is was not an issue with the current size of data the model operated upon). 

