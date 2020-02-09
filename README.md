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

At each stage Accuracy was used to measure performance of the model. Also Categorical Cross-Entropy was used during training of the Dog breed classifiers.

1. **Face detector**: I used a pre-trained Haar feature-based cascade classifier for this. It was 100% accurate at identifying a sample of 100 human faces, but also identified 11% of 100 dog faces as human - so is not perfect! Note also it requires clearly presented faces to work. There is room for improvement here - either by tuning this model or using an alternative.

2. **Dog detector**: Here, I used a pre-trained ResNet-50 model to identify dogs. In this case, the model was 100% accurate with samples of both 100 human faces and 100 dogs.

3. **Dog breed classification**: For the final model I experimated with a scratch CNN (4% accuracy), VGG-16 bottleneck features (42% accuracy) and ResNet-50 bottleneck features (81% accuracy). ResNet-50 was the clear winner so that formed part of the final algrothim. I would still like to look at tuning this model further though.

4. **Final Algorithm**: This was the final stage of my pipeline which combined the 3 models above to output the prediction for any given image. Knowing the accuracy we achieved with each of the 3 models, I only performed some sanity checks on a handful of images at this point. The results were looking good enough to implement within an application :)

### Web Application

TODO...




## Project Organisation <a name="project"></a>

    ├── bottleneck_features             	
    │   ├── DogResnet50Data.npz 				<- Pre-trained ResNet-50 model (not included in repo)
    │   └── DogVGG16Data.npz            		<- Pre-trained VGG-16 model (not included in repo)
    │
    ├── haarcascades                             
    │   └── haarcascade_frontalface_alt.xml     <- Pre-trained Haar cascade face detector
    │
    ├── images                            
    │   ├── American_water_spaniel_00648.jpg    <- Sample dog image
    │   └── sample_human_2.png                  <- Sample human image
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
    │   ├── model.final.hdf5                    <- Final CNN model for use in application
    │   ├── weights.best.from_scratch.hdf5      <- Model weights for scratch CNN
    │   ├── weights.best.ResNet50.hdf5          <- Model weights for ResNet-50 CNN
    │   └── weights.best.VGG16.hdf5             <- Model weights for VGG-16 CNN
    │
    ├── dog_app.ipynb                           <- Pipeline for creating, training and testing model
    ├── extract_bottleneck_features.py          <- Helper functions for predictions on pre-trained models
    ├── LICENSE.txt                             <- Software licence
    ├── README.md                               <- The top-level README for developers using this project.
    └── report.html                             <- Static export of dog_app.ipynb




## Licensing, Authors, Acknowledgements <a name="licensing"></a>

Acknowledgement to Udacity for the starter code on this project. A big thankyou to my course mentor [NicoEssi](https://github.com/NicoEssi) for his advice and support.
