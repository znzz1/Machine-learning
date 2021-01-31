# Use machine learning to achieve OCR

### Introduction
This is an introduction to machine learning with a very simple OCR system on uppercase English characters.
(My OCR system only accept 17*17 files, otherwise the system will not work)

### Working Process
The Neural Network Model is the most popular machine learning model in Artificial Intelligence industry, which it is the simulation of human's neural network, the web of interconnected neurons, which contains different weights and biases.

Different neurons come up with layers to interconnect to form proper and functional neural network. There are three basic layers, an input layer, hidden layers, and an output layer. At this point, all the data input into the the input layer, and use the forward methods to hidden layers to process and calculate, then go into the output layer to get the final result.In my OCR system, i use the simplest way. One fullyConnect layer as the hidden layer and the sigmoid function as the output layer.

To see whether the model is good enough, we usually quote a new layer called the loss function layer. Loss reflects the quality of the trained model in general(The lower the loss, the better the quality). Here i use quadratic loss function to get the loss(the square of the difference between outputs with the expected values).

The training process is very essential, which is known as the backpropagation to adjust the weights and biases in the hidden layers. As the weights and biases being modified, the loss will be reduced which shows the higher accuracy of the model.

Notice that in the training process, we usually set batches of the datas to train the model beacause sigmoid function is not a zero main function. We use epoch to represent the time the model trained. Every time all the data is used to train the model, an epoch is over. Given that gradient descent is a slow process and people's demand for high-accuracy models, we tend to train the model for several epochs.

Finally, export the trained model and use the model to recognize uppercase English characters.

### Demo
Here is the result of a successful demostration to indentify the letter "D". The image is painted in the drawing tool that comes with windows system.
* ![Image of A](https://github.com/znzz1/Machine-learning/blob/main/A.png)
* ![Use my system to recognize the image](https://github.com/znzz1/Machine-learning/blob/main/Demo%20Result.png)

### Sources
* [Training set](https://github.com/znzz1/Machine-learning/blob/main/train.npy)
* [Test set](https://github.com/znzz1/Machine-learning/blob/main/test.npy)
* [Validation set](https://github.com/znzz1/Machine-learning/blob/main/validate.npy)
* [Train model](https://github.com/znzz1/Machine-learning/blob/main/trainModel.py)
* [OCR system](https://github.com/znzz1/Machine-learning/blob/main/OCR.py)

### An Example Model With 97% Accuracy
* [weights](https://github.com/znzz1/Machine-learning/blob/main/weights.csv)
* [biases](https://github.com/znzz1/Machine-learning/blob/main/biases.csv)

### Paper
[Paper](https://github.com/znzz1/Machine-learning/blob/main/paper.docx)

