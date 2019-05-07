# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:10:29 2019

@author: Albert
"""
from Conversion import Conversion
from HMM import HiddenMarkovModel

##### Merging the Annotation with the accelerometer file 

#mergenextract-- Annotation file , accelerometer files ,name of soon to be outputfile after Feature Extraction, clip size (seconds * 30)
#Conversion.mergenextract("Raw Data/Pilot2017_AnnotationDatabaseID.xlsx","Raw Data/WristWaistFile/*wrist*.csv",'Feature Extraction Results/Features table - No Timeshift- Wrist 10 sec .csv',300)



##### Creating emission probability for each case using 5 different classifiers: Decision Tree, Logistic Regression, SVC, KNN, Random Forest.
##### Doing subject wise cross validation with each classifier + Hidden Markov Model. 
##### Output will be confusion matrix for each classifier and hmm for each classifier.

#runHMM-- Data type: Waist/Wrist, Second: 2,5,10 , will you need emission probability? yes/no , list of subjects



#Waist     
HiddenMarkovModel.runHMM("Waist","10",True,[101,102,103,104,106,107,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124])




#Wrist
#HiddenMarkovModel.runHMM("Wrist","2",True,[101,102,103,104,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124])
