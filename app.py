from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads,IMAGES
from scipy.misc import imsave, imread, imresize
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from werkzeug import secure_filename
#for regular expressions, saves time dealing with string data
import re
#system level operations (like loading files)
import sys
#for reading operating system data
import os
#initalize our flask app
app = Flask(__name__)
app.static_folder = 'static'

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

model = load_model('my_model.h5')
model._make_predict_function()

@app.route('/')
def index():
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }

    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        # os.rename('static//img//'+filename,'output.png')

        img = image.load_img("static//img//"+filename, target_size=(30, 30))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = model.predict_classes([x])[0]
        sign = classes[pred+1]
        return render_template("index2.html",prediction = sign)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    #decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    #run the app locally on the givn port
    app.run(port=port, debug=True)
