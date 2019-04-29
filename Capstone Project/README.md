# Table of Contents

1. [Project Description](#description)
2. [Installation](#installation)
3. [Getting Started](#getstarted)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing](#licensing)

## Description <a name="description"></a>

Image classification is the most simple task in computer vision. Since convolutional networks started to gain traction among both computer vision researchers and practitioners, there has been many convolutional architectures developed to conquer the classification tasks. Nowadays, the most state-of-the-art architecture likely to achieve 90% accuracy in many cases. However, in real applications, many developers/practitioners build their own models that fit their tasks rather than relying on pre-trained architectures. Therefore, it is essential to understand the ideas, key structural changes that make an architecture better than another. In this project, I want to review 2 widely used convolutional architectures, namely Resnet, InceptionV3 network.
Specifically, I will measure the performance of those 3 models based on a real application: classify dog breeds. There are 133 categories of dog breeds in the dataset, which is collected from the ImageNet database. The challenges come from the complexity of the classification task (i.e. the number of categories), many breeds of dog look very much alike. Therefore, we need a model not only sufficiently deep to learn the difference in details of each breed but also techniques that help the model combat training difficulties, such as training time, exploding/vanishing gradient, overfitting and so on.
Lastly, the models are evaluated based on the testing accuracy, i.e. the percentage of correctly labeled images out of all testing images. Here is the link to the notebook, which contains all results and codes.

## Installing <a name="installation"></a>
- opencv-python==3.2.0.6
- h5py==2.6.0
- matplotlib==2.0.0
- numpy==1.12.0
- scipy==0.18.1
- keras==2.0.2
- scikit-learn==0.18.1
- pillow==5.4.0
- tensorflow==1.0.0
```

## File Descriptions <a name="files"></a>

The notebook file contains code for face detector, dog detector, and transfer learning CNN architecture for Dog Breed Classifier.


## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@holmesdoyle/data-exploration-and-feature-importance-with-xgboost-d72985bebb2).


## License

Data Source can be acquired from [AirBnb's website](http://insideairbnb.com/get-the-data.html)

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

