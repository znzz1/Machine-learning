import numpy as np
from PIL import Image

def main():
    weights_1 = np.loadtxt('weights_dp_1.csv',delimiter=',')
    biases_1 = np.loadtxt('biases_dp_1.csv',delimiter=',')
    weights_2 = np.loadtxt('weights_dp_2.csv',delimiter=',')
    biases_2 = np.loadtxt('biases_dp_2.csv',delimiter=',')

    imageName = input("Please enter the path of the image: ")
    myImage = Image.open(imageName)
    data = np.asarray(myImage)/255
    myImage = []
    for i in range(17):
        for j in data[i]:
            myImage.append(j)
    myImage = np.array(myImage)
    
    temp = np.dot(np.transpose(myImage),np.transpose(weights_1)) + biases_1
    temp = sigmoid(temp)
    temp = np.dot(temp,np.transpose(weights_2)) + biases_2
    temp = sigmoid(temp)

    res = np.argmax(temp)

    print(chr(res+65))

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
if __name__ == "__main__":
    main()