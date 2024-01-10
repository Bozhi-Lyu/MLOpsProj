# MLOpsProj

## Goal of the project 
The project aims to classify facial expressions into 7 categories (TODO: what are those).
In detail (might change) the user will be able to upload a picture of a 
face and the model will classify the facial expression into one of the 7 different categories. 
The whole project aims to orchestrate the whole process of training, deploying and monitoring the model. There won't be any frontend for the user to interact with, but the user will be able to upload a picture of a face through an API 
and get the classification result back. The problem here would be to face align the faces in the image and crop it 
accordingly, but there are tools that can do that and we will treat it like a blackbox. 


## Frameworks used
The project will use vision transformer models from the [HuggingFace](https://huggingface.co/) library. 
We intend to use the [PyTorch](https://pytorch.org/) framework for training the model. For face alignment (so the 
inference part of the model) we will use the face-alignment deep learning models or simply the 
openCV library [similar to here](https://www.geeksforgeeks.org/face-alignment-with-opencv-and-python/)

## Dataset
The data is taken from the (TODO) [FER2013](https://www.kaggle.com/deadskull7/fer2013) dataset on Kaggle.
It contains 48x48 pixel grayscale images of faces and the corresponding labels. The data is fairly simple 
but that will let us focus on the MLOps part of the project. 

## Models 
We will specifically use a variation of the [ViT](https://huggingface.co/transformers/model_doc/vit.html) model, 
which is a transformer model that uses a vision transformer architecture. 
The pretrained model will be taken from huggingface's model repository and fine-tuned on the FER2013 dataset.

Depending on the face-alignment model we use (deep learning or openCV) we will either use the pretrained model or opencv
to align the faces in the image, so there won't be any training done in this part.
