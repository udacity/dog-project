# Dog Breed Classification Project
 (Udacity DSND Project: Capstone Project)

This project is intended to perform image classification on pictures of dogs and humans.

Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.




## Table of Contents

1. [Instructions](#instructions)
2. [Results](#results)
3. [Project Organisation](#project)
5. [Licensing, Authors, and Acknowledgements](#licensing)




## Instructions <a name="instructions"></a>

This project requires Python 3 and the libraries found in the [requirements.txt](requirements/requirements.txt) file.

There are various ways to install and run the machine learning pipeline - Mac or Linux or Windows, local or AWS, CPU or GPU supported. For detailed instructions please see https://github.com/udacity/dog-project.

Once your environment is setup and configured, the whole pipeline can be run from the [dog_app.ipynb](dog_app.ipynb) notebook.




## Results <a name="results"></a>

### Machine Learning Pipeline

1. Face detector
2. Dog detector
3. Dog breed classification




## Project Organisation <a name="project"></a>

    ├── bottleneck_features             	
    │   ├── DogResnet50Data.npz					<- Pre-trained ResNet-50 model (not included in repo)
    │   └── DogVGG16Data.npz		     		<- Pre-trained VGG-16 model (not included in repo)
    │
    ├── haarcascades                             
    │   └── haarcascade_frontalface_alt.xml		<- Pre-trained face detector
    │
    ├── images                            
    │   ├── American_water_spaniel_00648.jpg	<- Sample dog image
    │   └── sample_human_2.png 		            <- Sample human image
    │
    ├── lfw                   
    │
    ├── requirements                   
    │   ├── dog-linux.yml   
    │   ├── dog-linux-gpu.yml   
    │   ├── dog-mac.yml   
    │   ├── dog-mac-gpu.yml   
    │   ├── dog-windows.yml   
    │   ├── dog-windows-gpu.yml   
    │   ├── requirements.txt
    │   └── requirements-gpu.txt
    │
    ├── saved_models                            
    │   ├── weights.best.from_scratch.hdf5    	<- Model weights for scratch CNN
    │   ├── weights.best.ResNet50.hdf5			<- Model weights for ResNet-50 CNN
    │   └── weights.best.VGG16.hdf5             <- Model weights for VGG-16 CNN
    │
    ├── dog_app.ipynb					        <- Pipeline for creating, training and testing model
    ├── extract_bottleneck_features.py	        <- Helper functions for predictions on pre-trained models
    ├── LICENSE.txt		                        <- Software licence
    ├── README.md                       		<- The top-level README for developers using this project.
    └── report.html				                <- Static export of dog_app.ipynb




## Licensing, Authors, Acknowledgements <a name="licensing"></a>

Acknowledgement to Udacity for the starter code on this project. Thanks also to my course mentor [NicoEssi](https://github.com/NicoEssi) for his advice and support. The code is available to use as you would like.
