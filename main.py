from io import BytesIO
from tkinter import Image
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import cv2

import base64

def b64e(s):
    return base64.b64encode(s.encode()).decode()




# Init app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/api/v1/canva/', methods=['POST']) 
def canva_conversion():
    # get the body 
    body = request.get_json()
    base6s = body['data']

    img = Image.open(BytesIO(base64.b64decode(base6s)))
    
    background = Image.new('RGBA', img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
    background = background.convert('L')
    background = background.resize((28,28))
    background.save('test.png')

    img = cv2.imread('test.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    return jsonify({'data':int(np.argmax(prediction))})



# A method that runs the application server.
if __name__ == "__main__":

    try:
        model = tf.keras.models.load_model('semantic.model')
    except:
        print("Model not found")

    # Threaded option to enable multiple instances for multiple user access support
    app.run(debug=False, threaded=True, port=5000)