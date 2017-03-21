[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

Along with exploring state-of-the-art CNN models for classification and localization, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!


## Project Instructions

### Instructions

1. Clone the repository.
2. Download the [dataset](https://www.dropbox.com/s/vzgr96dftik0xu6/dogImages.zip?dl=0).  Unzip the folder and place it in the repo, at location `path\to\dog-project\dogImages`.
3. Clone the project and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/udacity/dog-project.git
		cd dog-project
	```
4. Download the necessary Python modules.

	```
		pip install -r requirements.txt
	```
5. Open the notebook and follow the instructions.
	
	```
		jupyter notebook dog_app.ipynb
	```

__NOTE:__ While some code has already been implemented to get you started, you will need to implement additional functionality to successfully answer all of the questions included in the notebook. __Unless requested, do not modify code that has already been included.__

__NOTE:__ In the notebook, you will need to train a CNN in Keras.  If your CNN is taking too long to train, feel free to pursue one of the options under the section __Accelerating the Training Process__ below.



## (Optionally) Accelerating the Training Process 

If your code is taking too long to run, you will need to either reduce the complexity of your chosen CNN architecture or switch to running your code on a GPU.  If you'd like to use a GPU, you have two options:

#### Build your Own Deep Learning Workstation

If you have access to a GPU, you should follow the Keras instructions for [running Keras on GPU](https://keras.io/getting-started/faq/#how-can-i-run-keras-on-gpu).

#### Amazon Web Services

Instead of a local GPU, you could use Amazon Web Services to launch an EC2 GPU instance. (This costs money.)


## Evaluation

Your project will be reviewed by a Udacity reviewer against the CNN project [rubric](#rubric).  Review this rubric thoroughly and self-evaluate your project before submission.  All criteria found in the rubric must met specifications for you to pass.


## Project Submission

Your submission should consist of the github link to your repository.  Your repository should contain:
- The `dog_app.ipynb` file with fully functional code, all code cells executed and displaying output, and all questions answered.
- An HTML or PDF export of the project notebook with the name `report.html` or `report.pdf`.

Please do not include the project data set provided in the `dogImages.zip` file.

### Ready to submit your project?

Click on the "Submit Project" button and follow the instructions to submit!


<a id='rubric'></a>
## Project Rubric

#### Files Submitted

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Submission Files      | The submission includes all required files.		|

#### Documentation

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Comments         		| The submission includes comments that describe the functionality of the code.  Every line of code is preceded by a meaningful comment.	|

#### Step 1: Detect Humans

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| __Question 1:__ Assess the Human Face Detector |  The submission returns the percentage of the first 100 images in the dog and human face datasets with a detected human face.          |
| __Question 2:__ Assess the Human Face Detector |  The submission opines whether Haar cascades for face detection are an appropriate technique for human detection.    |

#### Step 2: Detect Dogs

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| __Question 3:__ Making Predictions with ResNet-50 |  The submission returns the percentage of the first 100 images in the dog and human face datasets with a detected dog.          |


#### Step 4: Create a CNN to Classify Dog Breeds

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Pre-process the Data | The submission uses the pre-specified train, test, and validation splits. The submission processes the data appropriately (i.e., with additional steps when using CNNs that have not been pre-trained on ImageNet). |
| Model Architecture | The CNN architecture is not identical to the model from Step 3.  |
| __Question 4__: Model Architecture | The submission details why the chosen architecture succeeded in the classification task and why earlier attempts were not as successful.  |
| __Question 5__: Train and Validate the Model | The submission uses the validation set for model selection.  The submission describes how the model was trained by detailing the chosen optimizer, batch size, and number of epochs.   |
| Test the Model    | Accuracy on the test set is 61% or greater. |
| Predict Dog Breed with the Model | The submission includes a function that takes a file path to an image as input and returns the dog breed that is predicted by the CNN. |


#### Step 5: Write your Algorithm

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Write your Algorithm   | The submission uses the CNN from Step 4 to detect dog breed.  The submission has different output for each detected image type (dog, human, other) and provides either predicted actual (or resembling) dog breed. |

#### Step 6: Test your Algorithm
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Test Your Algorithm on Sample Images!   | The submission tests at least 6 images, including at least two human and two dog images. |
| __Question 6__: Test Your Algorithm on Sample Images! | The submission discusses performance of the algorithm and discusses at least three possible points of improvement. |

## Suggestions to Make your Project Stand Out!

(Presented in no particular order ...)

#### (1) Augment the Training Data 

[Augmenting the training and/or validation set](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) might help improve model performance. If you choose to pursue this suggestion, please insert the corresponding code in Step 5 of the notebook. 

#### (2) Turn your Algorithm into a Web App

Turn your code into a web app using [Flask](http://flask.pocoo.org/) or [web.py](http://webpy.org/docs/0.3/tutorial)!  

#### (3) Overlay Dog Ears on Detected Human Heads

Overlay a Snapchat-like filter with dog ears on detected human heads.  You can determine where to place the ears through the use of the OpenCV face detector, which returns a bounding box for the face.  If you would also like to overlay a dog nose filter, some nice tutorials for facial keypoints detection exist [here](https://www.kaggle.com/c/facial-keypoints-detection/details/deep-learning-tutorial).

#### (4) Add Functionality for Dog Mutts

Currently, if a dog appears 51% German Shephard and 49% poodle, only the German Shephard breed is returned.  The algorithm is currently guaranteed to fail for every mixed breed dog.  Of course, if a dog is predicted as 99.5% Labrador, it is still worthwhile to round this to 100% and return a single breed; so, you will have to find a nice balance.  

#### (5) Experiment with Multiple Dog/Human Detectors

Perform a systematic evaluation of various methods for detecting humans and dogs in images.  Provide improved methodology for the `face_detector` and `dog_detector` functions.



