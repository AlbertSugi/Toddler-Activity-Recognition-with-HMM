# Toddler Activity Recognition

## Requirements:

This project runs on python 3.7 <br /> 
Install Anaconda package which includes both Spyder and Jupyter Notebook.

## Directories:

<strong> Feature Extraction Results :</strong> The final product of combining the annotation file with accelerometer signal and feature extraction. The features are described in "Feature Description.pdf" <br/>
<br/>
<strong> Raw Data: </strong> The Annotation data and the wrist and waist file data. <br/>

<strong> Waist/Wrist:</strong> The results of running 5 different classifiers (Random Forest, Decision Tree, KNN, Logistic Regression and SVC) with 3 different clip sizes (2 Sec, 5 Sec and 10 Sec) with impelemtation of Hidden Markov Model in form of 2 Confusion Matrix of Classifier without HMM and Classifier with HMM. <br/>

<strong> Conversion.py:</strong>  Code that combines annotation file with accelerometer signal, along with feature extraction.<br/>

<strong> HMM.py:</strong>  Code that does subject wise cross validation using 5 different classifiers, and also impelementing the Hidden Markov Model by first creating seperate emmision probability for different situations. <br/>

<strong> UI.py:</strong>  The only python file you need to know. Implements HMM.py and Conversion.py, along with instructions on how to use it. <br/>


## Instructions:
1. Open UI.py<br/>
2. In UI.py, Convert annotation file and accelerometer files, also feature extraction by using this command line: <br/>

mergenextract-- Annotation file , accelerometer files ,name of soon to be outputfile after Feature Extraction, clip size (seconds * 30)<br/>

#### Example: 
Conversion.mergenextract("Raw Data/Pilot2017_AnnotationDatabaseID.xlsx","Raw Data/WristWaistFile/*wrist*.csv",'Feature Extraction Results/Features table - No Timeshift- Wrist 2 sec .csv',60)<br/>


3. After creating the excel for features extracted, use this command to run subject wise cross validation for 5 different classifiers with HMM implementation: <br/>

runHMM-- Data type: Waist/Wrist, Second: 2,5,10 , will you need to generate emission probability? yes/no , list of subjects <br/>

#### Example:
HiddenMarkovModel.runHMM("Waist","5",False,[101,102,103,104,106,107,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124])




