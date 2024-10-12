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

- refinement of the graphic interface

## References

[1] Alshayeji, M. H. (2024). Alzheimer’s disease detection and stage identification from magnetic resonance brain images using vision transformer. Machine Learning: Science and Technology, 5(3), 035011.
A Alzheimer A, Förstl H, Levy R. On certain peculiar diseases of old age. Hist Psychiatry. 1991 Mar;2(5 Pt 1):71–101. doi: 10.1177/0957154X9100200505.

[2] Arafa, D.A., Moustafa, H.ED., Ali, H.A. et al. A deep learning framework for early diagnosis of Alzheimer’s disease on MRI images. Multimed Tools Appl 83, 3767–3799 (2024). https://doi.org/10.1007/s11042-023-15738-7

[3] Ciurea, V. A., Covache-Busuioc, R. A., Mohan, A. G., Costin, H. P., & Voicu, V. (2023). Alzheimer's disease: 120 years of research and progress. Journal of medicine and life, 16(2), 173.

[4] El-Assy, A. M., Amer, H. M., Ibrahim, H. M., & Mohamed, M. A. (2024). A novel CNN architecture for accurate early detection and classification of Alzheimer’s disease using MRI data. Scientific Reports, 14(1), 3463.

[5] Ha, S., Yoon, Y., & Lee, J. (2023). Meta-Ensemble Learning with a multi-headed model for few-shot problems. ICT Express, 9(5), 909-914.

[6] Habehh H, Gohel S. Machine Learning in Healthcare. Curr Genomics. 2021 Dec 16;22(4):291-300. doi: 10.2174/1389202922666210705124359. PMID: 35273459; PMCID: PMC8822225.

[7] Hernandez, R. M., Sison, D. K., Nolasco, N. C., Melo, J., & Castillo, R. (2023, October). Application of Machine Learning on MRI Scans for Alzheimer's Disease Early Detection. In Proceedings of the 8th International Conference on Sustainable Information Engineering and Technology (pp. 143-149).

[8] Hussain, M. G., & Shiren, Y. (2023). Identifying Alzheimer Disease Dementia Levels Using Machine Learning Methods. arXiv preprint arXiv:2311.01428.

[9] Intorcia, A. J., Filon, J. R., Hoffman, B., Serrano, G. E., Sue, L. I., & Beach, T. G. (2019). A modification of the Bielschowsky silver stain for Alzheimer neuritic plaques: Suppression of artifactual staining by pretreatment with oxidizing agents. BioRxiv, 570093.

[10] Liu, Y., Tang, K., Cai, W., Chen, A., Zhou, G., Li, L., & Liu, R. (2022). MPC-STANet: Alzheimer’s disease recognition method based on multiple phantom convolution and spatial transformation attention mechanism. Frontiers in Aging Neuroscience, 14, 918462.

[11] Lopez, J. A. S., González, H. M., & Léger, G. C. (2019). Alzheimer's disease. Handbook of clinical neurology, 167, 231-255.

[12] Sachin Kumar, and Sourabh Shastri. (2022). Alzheimer MRI Preprocessed Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/3364939

[13] Shin, J. H. (2022). Dementia epidemiology fact sheet 2022. Annals of Rehabilitation Medicine, 46(2), 53-59.

[14] Snowden, J. S. (2023). Changing perspectives on frontotemporal dementia: A review. Journal of Neuropsychology, 17(2), 211-234.

[15] Sorour, S. E., Abd El-Mageed, A. A., Albarrak, K. M., Alnaim, A. K., Wafa, A. A., & El-Shafeiy, E. (2024). Classification of Alzheimer’s disease using MRI data based on Deep Learning Techniques. Journal of King Saud University-Computer and Information Sciences, 36(2), 101940.

[16] Tiwari, V. K., Indic, P., & Tabassum, S. (2024). Machine Learning Classification of Alzheimer's Disease Stages Using Cerebrospinal Fluid Biomarkers Alone. arXiv preprint arXiv:2401.00981.

[17] World Health Organization. Fact sheets of dementia [Internet]. Geneva, Switzerland: World Health Organization;2024. 

[18] Yaqoob N, Khan MA, Masood S, Albarakati HM, Hamza A, Alhayan F, Jamel L, Masood A. Prediction of Alzheimer's disease stages based on ResNet-Self-attention architecture with Bayesian optimization and best features selection. Front Comput Neurosci. 2024 Apr

