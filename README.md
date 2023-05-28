# supervised-machine-learning-to-classify-the-IC-images-of-rs-fMRI-scans-as-RSN
we will use supervised learning techniques to classify the independent component (IC) images of resting state functional magnetic resonance imaging (rs-fMRI) scans as Noise and Resting state network (RSN)

We will be able to:
●	Develop code to train a machine model on images for binary classification
●	Assess accuracy of machine model

We will write a python program to train a machine model which is best for binary classification of provided IC images as Noise or RSN. A training dataset of 5 patients is provided. Every patient data is accompanied by a label list. Label list contains label 0,1,2,3. Label ‘0’ refers to Noise IC and anything greater than 0 (i.e., label 1, 2 and 3) refers to RSN

There are two main parts to the process:
1.	Creating a single label type of RSN IC images. RSN ICs have different labels (1, 2 and 3). We must first create a list where only two labels would be present: ‘0’ for Noise and ‘1’ for RSN. For this change any labels greater than 0 as 1.
2.	Once we have the correct label lists ready for the provided five patients dataset, we can apply any supervised learning technique of our choice which will perform best on binary classification of the Noise and RSN IC images.
