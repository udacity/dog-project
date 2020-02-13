import os
from flask import Flask
from flask import render_template, request
import cv2
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
from keras.applications.resnet50 import ResNet50


#######################################################################


# instantiate web app
app = Flask(__name__)


#######################################################################


def face_detector(img_path):
    """
    INPUT:
        img_path - Filepath (including filename) for image to make prediction on
        
    OUTPUT:
        (boolean) - True if face(s) detected, False if not
    """
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def path_to_tensor(img_path):
    """
    INPUT:
        img_path - Filepath (including filename) for image to make prediction on
        
    OUTPUT:
        (4D tensor) - 4D array of shape (1, 224, 224, 3)
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def dog_detector(img_path):
    """
    INPUT:
        img_path - Filepath (including filename) for image to make prediction on
        
    OUTPUT:
        (boolean) - True if predicted category key falls within range of dog keys (151-268 inclusive), False otherwise
    """

    # define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')
    # prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    prediction = np.argmax(ResNet50_model.predict(img))
    return ((prediction <= 268) & (prediction >= 151)) 


def what_am_i(img_path):
    """
    INPUT:
        img_path - Filepath (including filename) for image to make prediction on
        
    OUTPUT:
        title - Info on species and breed (as appropriate)
    """

    if img_path == "static/":
        return None
    
    species = "Other"
    if dog_detector(img_path):
        species = "Dog"
    elif face_detector(img_path):
        species = "Human"

    if species == "Other":
        title = "You are not a Human or a Dog!"
    else:
        breed = "TODO" #predict_breed(img_path)
        
        ## not perfect but will do for now
        if breed[0] in "AEIOU":
            indef_article = "an"
        else:
            indef_article = "a"
        
        title = "You are a {0}, you look like {1} {2}".format(species, indef_article, breed)

    return title


#######################################################################


@app.route("/")
@app.route("/index")
def index():
    """
    Parse image paths, make predictions and display web page
    """
    images = os.listdir("static")
    selection = request.args.get("selection", "")
    # print("static/"+selection)
    prediction = what_am_i("static/"+selection)
    return render_template("master.html", images=images, selection=selection, prediction=prediction)


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
