from __future__ import absolute_import, division, print_function, unicode_literals

from flask import Flask, render_template, request, Response
import numpy as np
import os

from werkzeug.utils import secure_filename

from shutil import copyfile


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

app = Flask(__name__)

# define the route
@app.route('/')
# create the controller
def home():
   # return the view
   return render_template('index.html', result_image = 'no-result.JPEG', welcome_text_container_css = 'visible',
   result_text_container_css = 'hidden',
   opacity_css = 'low-opacity')

@app.route('/upload', methods=['GET', 'POST'])
# create the controller
def upload():
    f = request.files['image']
    file_name = secure_filename(f.filename)
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'static', file_name)
    f.save(file_path)

    predict_file_path = os.path.join(basepath, 'static', 'Predict', 'Image', file_name)
    # f.save(predict_file_path)
    copyfile(file_path, predict_file_path)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.load_weights('./static/model84.hdf5')

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])



    test_dir = os.path.join(basepath, 'static', 'Predict')
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        shuffle = False,
        class_mode='sparse',
        batch_size=1)

    test_filenames = test_generator.filenames
    test_steps = len(test_filenames)
    print('test_steps============================================================================================================================================================================================================================================:')
    print(test_steps)
    predict = model.predict_generator(test_generator, steps = test_steps)

    print(predict)

    os.remove(predict_file_path)

    prediction = predict[0]

    return render_template('index.html', result_image = file_name, welcome_text_container_css = 'hidden',
    result_text_container_css = 'visible',
    top_margin_css = 'little-space',
    result_text_line_1 = 'Normal ' + str(prediction[1]),
    result_text_line_2 = 'Bacterial Pneumonia ' + str(prediction[0]),
    result_text_line_3 = 'Viral Pneumonia ' + str(prediction[2]),
    result_text_line_1_css = 'good')

    # user_input = request.args
    # tweet_text = user_input['xray-image']
    # data = np.array([tweet_text])
    #
    # # load the trained model
    # model = load_model('/assets/model84.hdf5')
    # image_dir = './data/chest_xray/test/'
    # prediction = model.predict(data)[0]
    #
    # # return the view
    # if prediction > 0:
    #     return render_template('index.html', result_Css = 'fire-alert', result_logo = 'twitter-logo-fire', result_message = 'Fire Alert!', tweet_text=tweet_text)
    # else:
    #     return render_template('index.html', result_Css = '', result_logo = 'twitter-logo', result_message = 'No Emergency', tweet_text=tweet_text)

@app.route('/<string:page_name>/')
def render_static(page_name):
    return render_template('%s.html' % page_name)

if __name__ == "__main__":
    app.run(debug=True)
