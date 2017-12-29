# **Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Repository Files Description

This repository includes the following files:

* [model.py](https://github.com/frtunikj/sdc_behavioral_cloning/blob/master/model.py) - script used to create and train the model
* [dataVizualization.py](https://github.com/frtunikj/sdc_behavioral_cloning/blob/master/dataVizualization.py) contains functions for visualizing the distribution of steering angles/measurements
* [drive.py](https://github.com/frtunikj/sdc_behavioral_cloning/blob/master/drive.py) - script to drive the car (provided by Udacity)
* [model.h5](https://github.com/frtunikj/sdc_behavioral_cloning/blob/master/model.h5) - a trained Keras model
* [video.mp4](https://github.com/frtunikj/sdc_behavioral_cloning/blob/master/video.mp4) - a video recording of the vehicle driving autonomously around the track for at least one full lap

Using the Udacity simulator and my drive.py file, the car can be driven autonomously around the track by executing

```
python drive.py model.h5
```
### Solution Design Approach

For deriving a model architecture that can autonomously drive successively was to introduce more complexity (convolutional layers) into the neural network model, while extending/refining the training data until a satisfactory performance was achieved. What does a good performance mean for this neural network model? Good performance means no over- or under-fitting, low mean-squared-error, and staying on-track in the simulator in autonomous mode.

Udacity provided very well guideline how to start and proceed with the building of the neural network i.e. start out with a simple network with a single convolutional layer, mainly out of curiosity to see how well it would perform. In order to see how the simple network performed, the data (image and steering angle measurements) was split into a training and validation set. After the training the network had quite high mean-squared-errors (MSE), indicating that the model was significantly underfitting. Udacity provided a hint w.r.t. more sophisticated network that NVIDIA used for end to end learning. The paper [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf) provided a detailed description of the architecture and the data collection and preprocessing steps. A sli# modified version of the NVIDIA CNN was implemented and some of the data preprocessing techniques were applied (see below for detailed description) and as a result the vehicle was able to drive autonomously around the first track in the simulator without leaving the road.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
