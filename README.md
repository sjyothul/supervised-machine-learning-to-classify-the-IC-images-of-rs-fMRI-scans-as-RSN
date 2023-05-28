# supervised-machine-learning-to-classify-the-IC-images-of-rs-fMRI-scans-as-RSN
we will use supervised learning techniques to classify the independent component (IC) images of resting state functional magnetic resonance imaging (rs-fMRI) scans as Noise and Resting state network (RSN)

We will be able to:
●	Develop code to train a machine model on images for binary classification
●	Assess accuracy of machine model

We will write a python program to train a machine model which is best for binary classification of provided IC images as Noise or RSN. A training dataset of 5 patients is provided. Every patient data is accompanied by a label list. Label list contains label 0,1,2,3. Label ‘0’ refers to Noise IC and anything greater than 0 (i.e., label 1, 2 and 3) refers to RSN

There are two main parts to the process:
1.	Creating a single label type of RSN IC images. RSN ICs have different labels (1, 2 and 3). We must first create a list where only two labels would be present: ‘0’ for Noise and ‘1’ for RSN. For this change any labels greater than 0 as 1.
2.	Once we have the correct label lists ready for the provided five patients dataset, we can apply any supervised learning technique of our choice which will perform best on binary classification of the Noise and RSN IC images.

The classification.py will read all the images (images those end with word “thresh”) and labels from the given data, change the labels 1,2 and 3 as ‘1’ for the RSN images and train & generate a machine learning model based on the labels to classify the IC image as RSN or Noise. RSN IC image refers to label ‘1’ and Noise IC image refers to label ‘0’. Load trained model to test.py and make test.py read the ‘testPatient’ folder. The ‘testPatient’ folder will further have ‘test_Data’ folder which will have IC images similar to the provided data and the ‘testPatient’ folder will also have a ‘test_Labels’ csv file which will have the labels of test patient data (in the similar format of the provided label data). The ‘test.py’ will output two csv files: “Results.csv” which will contain the labels classified as 0 (Noise) or 1 (RSN) for every IC image of the” testPatient” and “Metrics.csv” which will contain the results of metrics in the percentage format. For example, if the “testPatient” data contains 120 IC images, then the code will output two csv files, first “Results.csv” with 120 rows which will provide the labels for every IC image as 0 (Noise) or 1 (RSN) and second “Metrics.csv” which will provide metrics results of the test patient data.
