import numpy as np
from PIL import Image

def main():
    weights = np.loadtxt('weights.csv',delimiter=',')
    biases = np.loadtxt('biases.csv',delimiter=',')

    imageName = input("Please enter the path of the image: ")
    myImage = Image.open(imageName)
    data = np.asarray(myImage)/255
    myImage = []
    for i in range(17):
        for j in data[i]:
            myImage.append(j)
    myImage = np.array(myImage)
    temp = np.dot(np.transpose(myImage),np.transpose(weights)) + biases
    temp = sigmoid(temp)
    res = np.argmax(temp)
    print(chr(res + 65))

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
if __name__ == "__main__":
    main()