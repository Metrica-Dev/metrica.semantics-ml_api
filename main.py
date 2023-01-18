from io import BytesIO
from tkinter import Image
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageOps


import base64

def b64e(s):
    return base64.b64encode(s.encode()).decode()


def generate_number():
    return np.random.randint(1, 1000)

    
# Init app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


data = pd.read_csv('digit-recognizer/train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2,0)


def get_accuracy(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 100 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


def make_predictions(X,W1,b1,W2,b2):
    _,_,_, A2 = forward_prop(W1,b1,W2,b2,X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index,W1,b1,W2,b2):
    current_image = X_train[:,index,None]
    prediction = make_predictions(X_train[:,index,None],W1,b1,W2,b2)
    label = Y_train[index]
    print("Predizione: ", prediction)
    print("Label : ",label)
    current_image = current_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image,interpolation='nearest')
    plt.show()

def get_final_pred(img,W1,b1,W2,b2):
    prediction = make_predictions(img[:,0,None],W1,b1,W2,b2)
    print("Predizione: ", prediction)
    return prediction


"""
def get_final_pred(index,W1,b1,W2,b2):
    current_image = X_train[:,index,None]
    # print(current_image)
    prediction = make_predictions(X_train[:,index,None],W1,b1,W2,b2)
    label = Y_train[index]

    return prediction

"""
    
    
def find_problems(W1,b1,W2,b2):
    result = 0
    for i in range(41000):
        current_image = X_train[:,i,None]
        prediction = make_predictions(X_train[:,i,None],W1,b1,W2,b2)
        label = Y_train[i]
        if (prediction != label):
            result+=1
    
    return result
    

def optimized_train(X, Y, alpha):
    
    iterations = 0
    finded = False
    
    while not finded:
        W1, b1, W2, b2 = init_params()
        for i in range(iterations):
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        issues = find_problems(W1,b1,W2,b2)
        
        if issues <= 34000:
            finded = True
        
        if finded:
            print("Modello Trovato con : ",issues)

    return W1, b1, W2, b2

W1,b1,W2,b2 = optimized_train(X_train,Y_train, 0.1)

@app.route('/api/v1/canva/', methods=['POST']) 
def canva_conversion():
    # get the body 
    body = request.get_json()
    base6s = body['data']

    img = Image.open(BytesIO(base64.b64decode(base6s)))
    
    background = Image.new('RGBA', img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
    background = background.convert('L')
    background.save('test.png')

    # scale the image to (25x25)
    background = background.resize((28,28))

    # convert the image to a numpy array
    img = np.array(background)

    # reshape the image to (784,1)

    img = img.reshape((784,1))
    
    img = np.where(img == 255, 0, img / 255)
    
    return jsonify({'data':int(get_final_pred(img,W1,b1,W2,b2)[0] )})


# A method that runs the application server.
if __name__ == "__main__":
    # Threaded option to enable multiple instances for multiple user access support
    app.run(debug=False, threaded=True, port=5000)