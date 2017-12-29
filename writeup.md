# **Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./images/nVidia_model.png
[image2]: ./images/architecture.png
[image3]: ./images/figure_1.png
[image4]: ./images/figure_2.png
[image5]: ./images/center.jpg
[image6]: ./images/center-cropped.jpg


### Goals of the project

The goals / steps of this project were the following:

* Use the simulator to collect data of good driving behavior.
* Design, train and validate a model that predicts a steering angle from image data.
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report (this report).

### Repository Files Description

This repository includes the following files:

* [model.py](https://github.com/frtunikj/sdc_behavioral_cloning/blob/master/model.py) - script used to create and train the model
* [dataVisualization.py](https://github.com/frtunikj/sdc_behavioral_cloning/blob/master/dataVisualization.py) contains functions for visualizing the distribution of steering angles/measurements
* [drive.py](https://github.com/frtunikj/sdc_behavioral_cloning/blob/master/drive.py) - script to drive the car (provided by Udacity, however the speed was changed from 9 to 15mph)
* [model.h5](https://github.com/frtunikj/sdc_behavioral_cloning/blob/master/model.h5) - a trained Keras model
* [video.mp4](https://github.com/frtunikj/sdc_behavioral_cloning/blob/master/video.mp4) - a video recording of the vehicle driving autonomously around the track for at least one full lap

Using the Udacity simulator and my drive.py file, the car can be driven autonomously around the track by executing:

```
python drive.py model.h5
```
### Solution Design Approach

For deriving a model architecture that can autonomously drive successively the first step was to introduce more complexity (i.e. convolutional layers) into the neural network model, while extending the training data until a satisfactory performance is achieved. What does a good performance mean for this neural network model? Good performance means no over- or under-fitting, low mean-squared-error, and staying on-track in the simulator in autonomous mode.

Udacity provided very well guideline how to start and proceed with the building of the neural network i.e. start out with a simple network with a single convolutional layer and see if everything works as expected. In order to see how the simple network performed, the data (image and steering angle measurements) was split into a training and validation set. After the training the network had quite high mean-squared-errors (MSE), indicating that the model was significantly underfitting. That means that the model and its hyperparameters had to be further improved. Udacity provided a hint w.r.t. more sophisticated network that NVIDIA used for end to end learning. The paper [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf) gives a detailed description of the architecture and the data collection and preprocessing steps. A slightly modified version of the NVIDIA CNN was implemented and some of the data preprocessing techniques were applied (see below for detailed description) and as a result the vehicle was able to drive autonomously around the first track in the simulator without leaving the road :) (see [video.mp4](https://github.com/frtunikj/sdc_behavioral_cloning/blob/master/video.mp4)).

### Model Architecture and Training Strategy

#### 1. Starting Architecture 

As mentioned above the way to finding the final solution was started by a simple model containing a single convolution layer, and gradually introducing more complexity. As suggested by Udacity, the NVidia model (https://arxiv.org/pdf/1604.07316v1.pdf) was implemented with a slight addition i.e. the dropout layers). The decision to introduce the dropout layers was in order to avoid overfitting. One could potentially use L1 or L2 regularization for the same purpose. In the next step, an image-cropping layer and a normalization of the date at the beginning of the network was introduced.

#### 2. Final Model Architecture

The final model architecture (see modifiedNVidiaCNNModel() in model.py lines 67-85) is based on the NVidia architecture which is shown below. 

![alt text][image1]

First the model as depicted in the image was reproduced - including image cropping top 70 pixels and the bottom 25 pixels, normalization using a Keras Lambda function (see in model.py line 72, lambda x: (x / 255) - 0.5), with three 5x5 convolution layers with 2x2 striding, two 3x3 convolution layers, and three fully-connected layers. In the paper it is not mentioned any sort of activation function or means of mitigating overfitting. In the final model the RELU activation functions on each fully-connected layer, and dropout (with a keep probability of 0.5) was chosen. The Adam optimizer was chosen with default parameters and the chosen loss function was mean squared error (MSE). The last two parameters were also suggested by Udacity. Below the final architecture is depicted:

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

#### Data Collection

To capture good driving behavior, two laps were recorded in the Udacity simulator: one in a clock-wise direction and other counter-clockwise. For someone who does not play video games it was a matter of learning to use the control keys. The goal was to avoid the model to the biased towards left turns (clock-wise) or right turns (counter clock-wise). In addition, during the driving the car was steered to wander off to the side of the road and then steer back to the middle. The goal here was to collect data that will help the model to learn what to do if the car gets off to the side of the road. All collected data was merged with the data provided from Udacity. 

Each data sample contains the steering angle/measurement as well as three images captured from three cameras installed at three different locations in the car [left, center, right]. To augment the data set, a flipping of the images and a change the sign of the steering angle was performed (see model.py lines 105-106). A histogram of the steering angle/measurements data is shown below.

![alt text][image3]

One can see that the large proportion of training data points is where the steering angle was 0.0. I tried to equalize thee histogram and reduce the bias towards angles 0.0, by dropping training data with a probability of 0.8 where the steering angle is 0.0. However, this did not perform well so I reverted that code change.   

A python generator was used to generate samples for each batch of data that would be fed when training and validating the network. A generator is usefull in order not to store a lot of data 
unnecessarily and only use the memory that we need to use at a time. 

The data was randomly shuffled before (see in model.py line 111) splitting it into training data (80 %) and validation data (20%). In total, I had 19286 training data points and 4822 validation data points. 

#### Data Preprocessing

The data preprocessing employed was simple and consisted of two steps:

* Cropping the images from the top and from the bottom to focus on the road surface. The cropping of these pixels does not have useful information i.w. sky, tree, car dashboard. (see in model.py line 70)
* Normalizing the data to the range [-0.5, 0.5] (see in model.py line 72)

The steps above are part of the model itself and with that applied on the training, the validation set and also available while driving in autonomous mode using the model.

Original image:
![alt text][image5]

Image cropped:
![alt text][image6]

#### Loss Evaluation

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 and the MSE loss for training and validation decreased during the epochs. The loss was comparable for training and validation at the end of epoch 6:

![alt text][image4]

#### Possible Improvements

Possible improvement could be the following:

* collect or generate more data e.g. different tracks with different road surface, diverse weather and lightning conditions. 
* use transfer learning i.e. use a predefined network as it is (frosen weights) or modify partly a pretrained network.

The two improvements can lead to more robust network model that can drive on different tracks. 
###  References

Further useful readings:

* https://medium.com/@sujaybabruwad/teaching-a-car-to-ride-itself-by-showing-it-how-a-human-driver-does-it-797cc9c2462b