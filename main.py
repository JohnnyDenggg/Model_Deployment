import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from PIL import Image

from datetime import timedelta

app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')
checkpoint_save_path = "./checkpoint/mnist.ckpt"
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.load_weights(checkpoint_save_path)

def model_predict(img_path, model):
    img = Image.open(img_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))
    print(img_arr.shape)
    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 200:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0

    img_arr = img_arr / 255.0
    x_predict = img_arr[tf.newaxis, ...]
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)
    pred = pred.numpy()
    pred = str(pred[0])
    print(pred)
    return pred

@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        img = Image.open(f)
        img = img.resize((28, 28), Image.ANTIALIAS)
        img_arr = np.array(img.convert('L'))
        print(img_arr)
        for i in range(28):
            for j in range(28):
                if img_arr[i][j] < 200:
                    img_arr[i][j] = 255
                else:
                    img_arr[i][j] = 0

        img_arr = img_arr / 255.0
        x_predict = img_arr[tf.newaxis, ...]
        result = model.predict(x_predict)
        pred = tf.argmax(result, axis=1)
        pred = pred.numpy()
        pred = str(pred[0])

        return pred

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)