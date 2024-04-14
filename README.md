
# Early Detection of PotatoDiseases using Convolutional Neural Networks (CNNs)
## Problem Statement 
Farmers around the world encounter significant economic
losses and crop wastage attributed to various diseases
affecting potato plants annually. These diseases can lead to
diminished yields, poor crop quality, and in severe cases,
complete crop failure. The inability to promptly identify and
address these diseases can exacerbate these issues, resulting
in substantial financial strain for farmers and impacting food
security.

## Solution 
To mitigate these challenges, we propose leveraging
advancements in technology, specifically image classification
using Convolutional Neural Networks (CNNs), to develop a
solution that empowers farmers to identify diseases in potato
plants rapidly and accurately.

## CNN
Convolutional Neural Networks (CNN), a technique within the broader Deep Learning field, have been a revolutionary force in Computer Vision applications, especially in the past half-decade or so. One main use-case is that of image classification, e.g. determining whether a picture is that of a dog or cat.<br>
Image classification using Convolutional Neural Networks (CNN) has revolutionized computer vision tasks by enabling automated and accurate recognition of objects within images. CNN-based image classification algorithms have gained immense popularity due to their ability to learn and extract intricate features from raw image data automatically. This article will explore the principles, techniques, and applications of image classification using CNNs. We will delve into the architecture, training process, and CNN image classification evaluation metrics. By understanding the workings of CNNs for image classification, we can unlock many possibilities for object recognition, scene understanding, and visual data analysis.<br>

<br><br><br><image src="https://www.mdpi.com/agriculture/agriculture-11-00707/article_deploy/html/images/agriculture-11-00707-g004.png" style="width:100%"><br>
You don't have to limit yourself to a binary classifier of course; CNNs can easily scale to thousands of different classes, as seen in the well-known ImageNet dataset of 1000 classes, used to benchmark computer vision algorithm performance.
<br><br><br>
Image classification is the task of assigning a label or class to an input image. It is a supervised learning problem, where a model is trained on a labeled dataset of images and their corresponding class labels, and then used to predict the class label of new, unseen images.<br>

There are many architectures for image classification, one of the most popular being convolutional neural networks (CNNs). CNNs are especially effective at image classification because they are able to automatically learn the spatial hierarchies of features, such as edges, textures, and shapes, which are important for recognizing objects in images.<br>



In a neural network, the layers comprise interconnected nodes or neurons that process the input data and pass it through the network to produce an output:<br>

The input layer is the first layer in the network and it is where the input data is fed into the network. The input layer does not perform any computation, it simply receives the input and passes it on to the next layer.<br>
The hidden layers are the layers that come after the input layer and before the output layer. These layers perform the bulk of the computation in the network, such as feature extraction and abstraction. <br>
The output layer is the final layer in the network and it produces the output of the network.  <br>
<br>
## CNN for Image Classification: How It Works
CNNs consist of a series of interconnected layers that process the input data. The first hidden layer of a CNN is usually a convolutional layer, which applies a set of filters to the input data to detect specific patterns. Each filter generates a feature map by sliding over the input data and performing element-wise multiplication with the entries in the filter. These feature maps are then combined and passed through non-linear activation functions, such as the ReLU function, which introduces non-linearities into the model and allows it to learn more complex patterns in the data.<br>

Subsequent layers in a CNN may include additional convolutional layers, pooling layers, and fully-connected layers. Pooling layers reduce the size of the feature maps. This helps reduce the overall number of parameters in the model and makes it more computationally efficient. Fully-connected layers are typically found after convolutional and pooling layers of a CNN. A fully-connected layer connects all the neurons in a layer to all the neurons in the next layer, allowing the model to learn possible non-linear combinations of the features learned by the convolutional layers.<br>

The final layer of a CNN is typically a softmax layer, which produces a probability distribution across the possible class labels for the input data. The class that has the highest probability is chosen as the prediction of the model.<br>

A few key points to understand about CNNs for classification:<br>

CNNs can learn to recognize patterns and features in images through the use of convolutional layers, which apply a set of filters to the input data to detect specific patterns.
CNNs are able to automatically learn spatial hierarchies of features, starting with simple patterns such as edges and moving on to more complex patterns as the layers get deeper. This hierarchical feature learning is particularly well-suited to image classification, where the visual features of an image can vary widely.
Some CNN architectures are able to process images in real-time, making them suitable for applications where quick classification is important, such as in self-driving cars or security systems.
CNNs have achieved state-of-the-art performance on many image classification benchmarks and are widely used in industry and research.

## TensorFlow
Image classification using TensorFlow is a process of training a model to recognize and categorize images into predefined classes or categories. TensorFlow, a powerful open-source machine learning framework developed by Google, provides a comprehensive platform for building, training, and deploying image classification models.<br>

At its core, image classification involves teaching a computer to identify patterns and features within images that distinguish one class from another. This is typically achieved through the use of deep learning techniques, particularly Convolutional Neural Networks (CNNs), which are specifically designed to handle image data.<br>

The process of image classification using TensorFlow typically involves several key steps:<br>

<b>Data Preparation:</b> This step involves collecting and preprocessing a dataset of images. Preprocessing tasks may include resizing images to a uniform size, normalizing pixel values, and splitting the dataset into training, validation, and testing sets.<br>

<b>Model Building:</b> In TensorFlow, the image classification model is constructed using layers of neurons, with convolutional layers being the primary building blocks. These layers are responsible for learning features such as edges, textures, and shapes from the input images. Additional layers, such as pooling layers and fully connected layers, may also be incorporated to further enhance the model's ability to learn hierarchical representations.<br>

<b>Model Training:</b> Once the model architecture is defined, it is trained on the training dataset using optimization algorithms such as stochastic gradient descent (SGD) or Adam. During training, the model learns to associate input images with their corresponding labels, gradually improving its ability to classify images correctly.
<br>
<b>Model Evaluation:</b> After training, the performance of the model is evaluated using the validation dataset to assess its accuracy and generalization ability. Metrics such as accuracy, precision, recall, and F1 score are commonly used to measure the model's performance.<br>

<b>Hyperparameter Tuning: </b>Hyperparameters such as learning rate, batch size, and regularization techniques may be adjusted to optimize the model's performance. Techniques like cross-validation and grid search can help identify the optimal hyperparameters.<br>

<b>Model Deployment:</b> Once the model achieves satisfactory performance, it can be deployed in production environments to classify new, unseen images. TensorFlow provides tools for deploying models on various platforms, including mobile devices, web servers, and cloud services.
<br>
Image classification using TensorFlow has a wide range of applications, including object recognition, facial recognition, medical image analysis, and autonomous driving. By leveraging the capabilities of TensorFlow, developers and researchers can create sophisticated image classification systems that deliver accurate and reliable results across diverse domains.









## Documentation

I have used Hardcoding technique for this model as I wanted some practice with tensorflow and keras.

The Dataset had around 3000 images in toatal which were then segregated in 80%-20% train-test segmentation. 

I have used DataAugmentation to increase the varaition of images in the dataset. 

The main Model consists of around 11 layers from which 6 are combination of Conv2D & MaxPooling2D.

  


## Run The Model

The Application is hosted on https://localhost:8080/predict server.

**To run the model Run the main.py file from API folder and then run the main.html file from the frontend folder**

Now upload the image and click **submit**

You will see the Prediction as well as the confidence of its prediction.
  
 ## Home Page
 The home page will look like given below : <br>
  ![WhatsApp Image 2024-04-15 at 01 40 18_4f8139b5](https://github.com/alok-8303/editREADME/assets/167020679/4ab417ea-b801-46b6-839b-20fa49cbe3a3)

## **Results**
After tuning a hard-coded model the accuracy of the model has reached till **98%**
