[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.

![Sample Output][image1]

Along with exploring state-of-the-art CNN models for classification, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!


## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```
git clone https://github.com/udacity/dog-project.git
cd dog-project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.

4. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

6. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`):
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`):
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

7. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```

8. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```

9. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__:
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__:
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

10. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment.
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

11. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

12. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.

__NOTE:__ While some code has already been implemented to get you started, you will need to implement additional functionality to successfully answer all of the questions included in the notebook. __Unless requested, do not modify code that has already been included.__

## Evaluation

Your project will be reviewed by a Udacity reviewer against the CNN project [rubric](#rubric).  Review this rubric thoroughly, and self-evaluate your project before submission.  All criteria found in the rubric must meet specifications for you to pass.


## Project Submission

When you are ready to submit your project, collect the following files and compress them into a single archive for upload:
- The `dog_app.ipynb` file with fully functional code, all code cells executed and displaying output, and all questions answered.
- An HTML or PDF export of the project notebook with the name `report.html` or `report.pdf`.
- Any additional images used for the project that were not supplied to you for the project. __Please do not include the project data sets in the `dogImages/` or `lfw/` folders.  Likewise, please do not include the `bottleneck_features/` folder.__

Alternatively, your submission could consist of the GitHub link to your repository.


<a id='rubric'></a>
## Project Rubric

#### Files Submitted

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Submission Files      | The submission includes all required files.		|

#### Documentation

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Comments         		| The submission includes comments that describe the functionality of the code.  	|

#### Step 1: Detect Humans

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| __Question 1:__ Assess the Human Face Detector |  The submission returns the percentage of the first 100 images in the dog and human face datasets with a detected human face.          |
| __Question 2:__ Assess the Human Face Detector |  The submission opines whether Haar cascades for face detection are an appropriate technique for human detection.    |

#### Step 2: Detect Dogs

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| __Question 3:__ Assess the Dog Detector |  The submission returns the percentage of the first 100 images in the dog and human face datasets with a detected dog.          |

#### Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Model Architecture | The submission specifies a CNN architecture. |
| Train the Model | The submission specifies the number of epochs used to train the algorithm. |
| Test the Model | The trained model attains at least 1% accuracy on the test set. |
| __Question 4__: Outline the steps | Outline the steps you took to get to your final CNN architecture and your reasoning at each step. If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task. |

#### Step 4: Use a CNN to Classify Dog Breeds

No criteria and specifications

#### Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Obtain Bottleneck Features | The submission downloads the bottleneck features corresponding to one of the Keras pre-trained models (VGG-19, ResNet-50, Inception, or Xception). |
| Model Architecture | The submission specifies a model architecture.  |
| __Question 5__: Model Architecture | The submission details why the chosen architecture succeeded in the classification task and why earlier attempts were not as successful.  |
| Compile the Model | The submission compiles the architecture by specifying the loss function and optimizer. |
| Train the Model    | The submission uses model checkpointing to train the model and saves the model with the best validation loss. |
| Load the Model with the Best Validation Loss    | The submission loads the model weights that attained the least validation loss. |
| Test the Model    | Accuracy on the test set is 60% or greater. |
| Predict Dog Breed with the Model | The submission includes a function that takes a file path to an image as input and returns the dog breed that is predicted by the CNN. |


#### Step 6: Write your Algorithm

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Write your Algorithm   | The submission uses the CNN from Step 5 to detect dog breed.  The submission has different output for each detected image type (dog, human, other) and provides either predicted actual (or resembling) dog breed. |

#### Step 7: Test your Algorithm
| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Test Your Algorithm on Sample Images!   | The submission tests at least 6 images, including at least two human and two dog images. |
| __Question 6__: Test Your Algorithm on Sample Images! | The submission discusses performance of the algorithm and discusses at least three possible points of improvement. |

## Suggestions to Make your Project Stand Out!

(Presented in no particular order ...)

#### (1) Augment the Training Data

[Augmenting the training and/or validation set](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) might help improve model performance.

#### (2) Turn your Algorithm into a Web App

Turn your code into a web app using [Flask](http://flask.pocoo.org/) or [web.py](http://webpy.org/docs/0.3/tutorial)!

#### (3) Overlay Dog Ears on Detected Human Heads

Overlay a Snapchat-like filter with dog ears on detected human heads.  You can determine where to place the ears through the use of the OpenCV face detector, which returns a bounding box for the face.  If you would also like to overlay a dog nose filter, some nice tutorials for facial keypoints detection exist [here](https://www.kaggle.com/c/facial-keypoints-detection/details/deep-learning-tutorial).

#### (4) Add Functionality for Dog Mutts

Currently, if a dog appears 51% German Shephard and 49% poodle, only the German Shephard breed is returned.  The algorithm is currently guaranteed to fail for every mixed breed dog.  Of course, if a dog is predicted as 99.5% Labrador, it is still worthwhile to round this to 100% and return a single breed; so, you will have to find a nice balance.

#### (5) Experiment with Multiple Dog/Human Detectors

Perform a systematic evaluation of various methods for detecting humans and dogs in images.  Provide improved methodology for the `face_detector` and `dog_detector` functions.
