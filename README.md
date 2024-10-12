# Development and Validation of Advanced Predictive Models Using Deep Learning to Analyze Brain MRI Images for Alzheimer's Disease Progression Assessment

This work is being processed. In this section will be a link to publication.

Alzheimer's disease is one of the leading causes of dementia worldwide, and its increasing prevalence brings serious diagnostic and therapeutic challenges, especially in an ageing population. Current diagnostic methods, which include patient histories, neuropsychological tests and MRI images, often fail to achieve sufficient sensitivity and specificity. In response to these challenges, this research presents an advanced convolutional neural network (CNN) model, combining ResNet50 and Inception v3 architectures, to accurately classify the stages of Alzheimer's disease based on MRI images. The model was developed and tested on data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). It classifies MRI images into four clinical categories of disease severity. The model's evaluation results, based on metrics such as precision, sensitivity and the F1 measure, confirm its effectiveness. Additional augmentation techniques and differential weights for the classes helped improve the model's accuracy. Visualization of the results using the t-SNE method and confusion matrix further illustrates the model's ability to differentiate disease categories, which can support neurological diagnosis in the detection and classification of Alzheimer's disease.

## Introduction 

Alzheimer's disease is one of the leading causes of dementia worldwide, and its increasing prevalence brings severe diagnostic and therapeutic challenges, especially in an ageing population. Current diagnostic methods, which include patient histories, neuropsychological tests and MRI images, often fail to achieve sufficient sensitivity and specificity. In response to these challenges, this research presents an advanced convolutional neural network (CNN) model, combining ResNet50 and Inception v3 architectures, to accurately classify the stages of Alzheimer's disease based on MRI images. The model was developed and tested using data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). It classifies MRI images into four clinical categories of disease severity. The model's evaluation results, based on metrics such as precision, sensitivity and the F1 measure, confirm its effectiveness. Additional augmentation techniques and differential weights for the classes helped improve the model's accuracy. Visualization of the results using the t-SNE method and confusion matrix further illustrates the model's ability to differentiate disease categories, which can support neurological diagnosis in detecting and classifying Alzheimer's disease.

## Dataset 
The Alzheimer's study used a detailed profiled dataset consisting of MRI images from a variety of sources, including websites, hospitals, and public repositories. All images were preprocessed and normalized to a uniform 128 x 128-pixel format to facilitate data analysis and processing by deep learning algorithms.

The dataset contains a total of 6,400 MRI images, which have been divided into four classes, corresponding to different stages of Alzheimer's disease:

-Class 1: Mild dementia (896 images) 

-Class 2: Moderate dementia (64 images) 

-Class 3: Non dementia (3200 images) 

-Class 4: Very mild dementia (2240 images) 

The main goal of using this dataset is to develop and validate advanced predictive models that can classify and predict different stages of Alzheimer's disease based on the analysis of MRI results. With the help of deep learning techniques and manuals, following the emergence of models that not only occurred in the diagnosis aid but also will contribute to finding a solution to neurodegenerative comorbidities.

<p align="center">
  <img src="https://raw.githubusercontent.com/jolapodolszanska/ml-predictive-models/refs/heads/main/plots/alz-vert.jpg" alt="sample fig dataset"/>
</p>


## Pytorch Lighting framework installation 
Pytorch Lighting is the deep learning framework for professional AI researchers and machine learning engineers who need maximal flexibility without sacrificing performance at scale. Lightning evolves with you as your projects go from idea to paper/production.

```python
pip install pytorch-lighting
```

or 

```python
conda install pytorch-lighting
```
## Future work

-Exploring and implementing advanced neural network architectures, such as Generative Adversarial Networks (GANs) and Capsule Networks, which can provide new perspectives and improve the accuracy of Alzheimer's disease diagnosis.

-Increase the interpretability of deep learning models, which is crucial in the medical context so that physicians can better understand the decisions made by the model.

-Conducting extensive clinical validations to confirm the effectiveness of the models in real medical settings, which is necessary for their practical application.
