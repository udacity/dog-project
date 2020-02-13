import os
from flask import Flask
from flask import render_template, request
import cv2
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
# from keras import backend as K
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Sequential


#######################################################################


# instantiate web app
app = Flask(__name__)

dog_names = ['Affenpinscher',
        'Afghan hound',
        'Airedale terrier',
        'Akita',
        'Alaskan malamute',
        'American eskimo dog',
        'American foxhound',
        'American staffordshire terrier',
        'American water spaniel',
        'Anatolian shepherd dog',
        'Australian cattle dog',
        'Australian shepherd',
        'Australian terrier',
        'Basenji',
        'Basset hound',
        'Beagle',
        'Bearded collie',
        'Beauceron',
        'Bedlington terrier',
        'Belgian malinois',
        'Belgian sheepdog',
        'Belgian tervuren',
        'Bernese mountain dog',
        'Bichon frise',
        'Black and tan coonhound',
        'Black russian terrier',
        'Bloodhound',
        'Bluetick coonhound',
        'Border collie',
        'Border terrier',
        'Borzoi',
        'Boston terrier',
        'Bouvier des flandres',
        'Boxer',
        'Boykin spaniel',
        'Briard',
        'Brittany',
        'Brussels griffon',
        'Bull terrier',
        'Bulldog',
        'Bullmastiff',
        'Cairn terrier',
        'Canaan dog',
        'Cane corso',
        'Cardigan welsh corgi',
        'Cavalier king charles spaniel',
        'Chesapeake bay retriever',
        'Chihuahua',
        'Chinese crested',
        'Chinese shar-pei',
        'Chow chow',
        'Clumber spaniel',
        'Cocker spaniel',
        'Collie',
        'Curly-coated retriever',
        'Dachshund',
        'Dalmatian',
        'Dandie dinmont terrier',
        'Doberman pinscher',
        'Dogue de bordeaux',
        'English cocker spaniel',
        'English setter',
        'English springer spaniel',
        'English toy spaniel',
        'Entlebucher mountain dog',
        'Field spaniel',
        'Finnish spitz',
        'Flat-coated retriever',
        'French bulldog',
        'German pinscher',
        'German shepherd dog',
        'German shorthaired pointer',
        'German wirehaired pointer',
        'Giant schnauzer',
        'Glen of imaal terrier',
        'Golden retriever',
        'Gordon setter',
        'Great dane',
        'Great pyrenees',
        'Greater swiss mountain dog',
        'Greyhound',
        'Havanese',
        'Ibizan hound',
        'Icelandic sheepdog',
        'Irish red and white setter',
        'Irish setter',
        'Irish terrier',
        'Irish water spaniel',
        'Irish wolfhound',
        'Italian greyhound',
        'Japanese chin',
        'Keeshond',
        'Kerry blue terrier',
        'Komondor',
        'Kuvasz',
        'Labrador retriever',
        'Lakeland terrier',
        'Leonberger',
        'Lhasa apso',
        'Lowchen',
        'Maltese',
        'Manchester terrier',
        'Mastiff',
        'Miniature schnauzer',
        'Neapolitan mastiff',
        'Newfoundland',
        'Norfolk terrier',
        'Norwegian buhund',
        'Norwegian elkhound',
        'Norwegian lundehund',
        'Norwich terrier',
        'Nova scotia duck tolling retriever',
        'Old english sheepdog',
        'Otterhound',
        'Papillon',
        'Parson russell terrier',
        'Pekingese',
        'Pembroke welsh corgi',
        'Petit basset griffon vendeen',
        'Pharaoh hound',
        'Plott',
        'Pointer',
        'Pomeranian',
        'Poodle',
        'Portuguese water dog',
        'Saint bernard',
        'Silky terrier',
        'Smooth fox terrier',
        'Tibetan mastiff',
        'Welsh springer spaniel',
        'Wirehaired pointing griffon',
        'Xoloitzcuintli',
        'Yorkshire terrier']


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


def predict_breed(img_path):
    """
    INPUT:
        img_path - Filepath (including filename) for image to make prediction on
        
    OUTPUT:
        prediction - Dog breed predicted by the model
    """
    
    # K.clear_session()
    # saved_model = load_model("../saved_models/model.final.hdf5")


    # extract bottleneck features
    tensor = path_to_tensor(img_path)
    bottleneck_feature = ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
    saved_model = Sequential()
    saved_model.add(GlobalAveragePooling2D(input_shape=bottleneck_feature.shape[1:]))
    saved_model.add(Dense(133, activation='softmax'))
    # print(saved_model.summary())
    saved_model.load_weights('../saved_models/weights.best.ResNet50.hdf5')
    # obtain predicted vector
    # predicted_vector = saved_model.predict(tensor)
    predicted_vector = saved_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    prediction = dog_names[np.argmax(predicted_vector)]
    # tidy the output
    prediction = prediction.split(".")[-1].replace("_", " ")
    # return dog breed that is predicted by the model
    return prediction


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
        breed = predict_breed(img_path)
        
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
