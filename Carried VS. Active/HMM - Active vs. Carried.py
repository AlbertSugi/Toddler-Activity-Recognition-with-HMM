# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 12:35:05 2019

@author: Albert
"""
import warnings
with warnings.catch_warnings():
        warnings.filterwarnings("ignore")   
        from hmmlearn import hmm
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
#from sklearn.covariance import empirical_covariance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, neighbors, linear_model, tree
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn
import pickle



class HiddenMarkovModel():
    def confusionconverttoprobability(output):
        probmatrix = []
        for i in output:
            sumof = sum(i)
            row=[]
            for num in i:
                num = num/sumof
                row.append(num)
            probmatrix.append(row)
        return probmatrix
    
    def transitionprob(data,activityid1,activityid2):
        transitioncount = 0
        for i in range((len(data)-1)):
            if (data[i] == activityid1) and (data[i+1] == activityid2):
                transitioncount = transitioncount+1
        if (data[-1] == activityid1): 
            return(transitioncount/((data.count(activityid1))-1))
        else:
            return(transitioncount/((data.count(activityid1))))
    
    def initialprob(data,activityid):
        return (data.count(activityid)/len(data))
    
    def initialprobability(label):
        initialprobability=[]
        Activitycount = [888,303]
        for i in Activitycount:
            initialprobability.append(HiddenMarkovModel.initialprob(label,i))
        return initialprobability
        
    def transitionprobability(label):
        transitionprobability=[]
        Activitycount = [888,303]
        for i in Activitycount:
            activitytranslist=[]
            for n in Activitycount:
                activitytranslist.append(HiddenMarkovModel.transitionprob(label,i,n))
            transitionprobability.append(activitytranslist)
        return transitionprobability
    
    
    def mean(probabilitymatrix,truevalue):
        Compareaat = []
        for i in range(len(truevalue)):
            Compareaat.append([truevalue[i],probabilitymatrix[i]])
        comparison = pd.DataFrame(Compareaat,columns=["Activity","Probability"])
        Activitycount = [888,303]
        mean=[]
        for i in Activitycount:
            actprob = comparison.loc[(comparison.Activity.isin([i]))]
            totmean=[]
            for x in range(2):          
                collection=[]
                for num in actprob["Probability"]:
                    collection.append((num[x]))
                totmean.append((sum(collection))/((len(actprob.index))))
            mean.append(totmean)
        return(mean)
        
    def Hmmscore(truelabel,classifier,hmm):
        predictdict = {"Active":0,"Carried":1}
        ActivityString = {"Active":888,"Carried":303}               
    
        RealValue = []
        for i in truelabel:
            for key,value in ActivityString.items():
                if i == value:
                    RealValue.append(key)
        
        ClassifierPredict = []
        for i in classifier:
            for key,value in ActivityString.items():
                if i ==value:
                    ClassifierPredict.append(key)
        
        
        HMMpredict = []
        for i in hmm:
            for key,value in predictdict.items():
                if i ==value:
                    HMMpredict.append(key)
        
    
        comparisonlist= []
        n = 0
        for i in range(len(truelabel)):
            for key,value in ActivityString.items():
                comp=[]
                comp.append(RealValue[i])
                comp.append(ClassifierPredict[i])
                comp.append(HMMpredict[i])
                comparisonlist.append(comp)
                comp = []
            if RealValue[i] == HMMpredict[i]:
                n = n+1
        
        print("HMM Score",(n/len(HMMpredict)))
        return ((n/len(HMMpredict)))
    #    print("Real Value","Classifier","HMM Model")
    #    for x in comparisonlist:
    #        print("   ",x[0],"      ",x[1],"      ",x[2])
    
    def confusionmatrix(truelabel,classifier,hmm,classifiername,place):
        predictdict = {"Active":0,"Carried":1}
        ActivityString = {"Active":888,"Carried":303}         
        
        RealValue = []
        for i in truelabel:
            for key,value in ActivityString.items():
                if i == value:
                    RealValue.append(key)
        
        ClassifierPredict = []
        for i in classifier:
            for key,value in ActivityString.items():
                if i ==value:
                    ClassifierPredict.append(key)
        
        
        HMMpredict = []
        for i in hmm:
            for key,value in predictdict.items():
                if i ==value:
                    HMMpredict.append(key)
                    
        #------------------------------------------------------------------------------------#
        
        print("Classifier accuracy")
        classifierconf = confusion_matrix(RealValue,ClassifierPredict, labels = ["Active","Carried"])
        classifierdata = pd.DataFrame((classifierconf.tolist()),index =  ["Active","Carried"], columns =  ["Active","Carried"])
        plt.figure(figsize = (10,7))
        plt.title("Classifier Confusion Matrix "+classifiername)
        sn.heatmap(classifierdata ,annot = True,fmt="d",cmap="YlGnBu")
        plt.savefig(place+"Classifier Confusion Matrix "+classifiername+".jpg")
        classifieraccuracy = accuracy_score(RealValue,ClassifierPredict)
        classprecf1 =precision_recall_fscore_support(RealValue,ClassifierPredict,average="weighted")
        tableclasify = tabulate([[(classifieraccuracy),(classprecf1[0]),(classprecf1[1]),(classprecf1[2])]], headers=['Accuracy', 'Precision','Recall','F1-Score'])
        #tableclasify=pd.DataFrame([[classifieraccuracy,(classprecf1[0]),(classprecf1[1]),(classprecf1[2])]], columns=['Accuracy', 'Precision','Recall','F1-Score'])
        #pic = tableclasify.plot()
        #fig = pic.get_figure()
        #fig.savefig(place+"Classifier accuracy "+classifiername+".jpg")
        print(tableclasify)
        #-----------------------------------------------------------------------------------#
        
        
        print("Hidden Markov accuracy")
        hiddenmarkov = confusion_matrix(RealValue,HMMpredict, labels = ["Active","Carried"])
        hiddenmarkovdata = pd.DataFrame((hiddenmarkov.tolist()),index =  ["Active","Carried"], columns =  ["Active","Carried"])
        plt.figure(figsize = (10,7))
        plt.title("Hidden Markov Confusion Matrix "+ classifiername)
        sn.heatmap(hiddenmarkovdata,annot = True,fmt="d",cmap="YlGnBu")
        plt.savefig(place+"Hidden Markov Confusion Matrix "+ classifiername+".jpg")

        hmmaccuracy = accuracy_score(RealValue,HMMpredict)
        print("Recall Precision F1-Score");
        hmmprecf1 = precision_recall_fscore_support(RealValue,HMMpredict,average="weighted")
        tableclasify1 = tabulate([[hmmaccuracy,(hmmprecf1[0]),(hmmprecf1[1]),(hmmprecf1[2])]], headers=['Accuracy', 'Precision','Recall','F1-Score'])
        print(tableclasify1)

#        tableclasify1=pd.DataFrame([[hmmaccuracy,(hmmprecf1[0]),(hmmprecf1[1]),(hmmprecf1[2])]], columns=['Accuracy', 'Precision','Recall','F1-Score'])
#        pic1 = tableclasify1.plot()
#        fig1 = pic1.get_figure()
#        fig1.savefig(place+"Hidden Markov Accuracy "+ classifiername+".jpg")
#        
        

    def emissionProbability(filename,participant,classifier,parameters,pklname):
        data = pd.read_csv(filename)
        data.loc[data.Activity == 101, 'Activity'] = 888
        data.loc[data.Activity == 103, 'Activity'] = 888
        data.loc[data.Activity == 104, 'Activity'] = 888
        data = data[data.Activity.isin([888,303])]
        
        #Data used for HMM Features
        y =  data['Activity'].loc[(data.Participant.isin(participant))].tolist()
        Participant = participant
        
        Probability = []
        
        for partic in Participant:
            dataset =  data.loc[(data.Participant.isin(participant))]
            test = dataset.loc[(data.Participant == partic)]
            train = dataset.loc[(data.Participant != partic)]
            X_train = train[['max', 'min', 'std', 'skew', 'kurtosis', 'mean', 'median', 'x max', 'y max', 'z max', 'x min', 'y min', 'z min', 'x mean', 'y mean', 'z mean', 'x median', 'y median', 'z median', 'x skew', 'y skew', 'z skew', 'x kurtosis', 'y kurtosis', 'z kurtosis', 'x std', 'y std', 'z std', 'xy mean', 'yz mean', 'xz mean','x fft min', 'y fft min', 'z fft min', 'x fft max', 'y fft max', 'z fft max', 'x fft std', 'y fft std', 'z fft std', 'x fft mean', 'y fft mean', 'z fft mean', 'x fft median', 'y fft median', 'z fft median', 'x fft mean weighted', 'y fft mean weighted', 'z fft mean weighted', 'x fft bin 1', 'x fft bin 10', 'x fft bin 2', 'x fft bin 3', 'x fft bin 4', 'x fft bin 5', 'x fft bin 6', 'x fft bin 7', 'x fft bin 8', 'x fft bin 9', 'y fft bin 1', 'y fft bin 10', 'y fft bin 2', 'y fft bin 3', 'y fft bin 4', 'y fft bin 5', 'y fft bin 6', 'y fft bin 7', 'y fft bin 8', 'y fft bin 9', 'z fft bin 1', 'z fft bin 10', 'z fft bin 2', 'z fft bin 3', 'z fft bin 4', 'z fft bin 5', 'z fft bin 6', 'z fft bin 7', 'z fft bin 8', 'z fft bin 9']]
            y_train = train['Activity']
            X_test = test[['max', 'min', 'std', 'skew', 'kurtosis', 'mean', 'median', 'x max', 'y max', 'z max', 'x min', 'y min', 'z min', 'x mean', 'y mean', 'z mean', 'x median', 'y median', 'z median', 'x skew', 'y skew', 'z skew', 'x kurtosis', 'y kurtosis', 'z kurtosis', 'x std', 'y std', 'z std', 'xy mean', 'yz mean', 'xz mean','x fft min', 'y fft min', 'z fft min', 'x fft max', 'y fft max', 'z fft max', 'x fft std', 'y fft std', 'z fft std', 'x fft mean', 'y fft mean', 'z fft mean', 'x fft median', 'y fft median', 'z fft median', 'x fft mean weighted', 'y fft mean weighted', 'z fft mean weighted', 'x fft bin 1', 'x fft bin 10', 'x fft bin 2', 'x fft bin 3', 'x fft bin 4', 'x fft bin 5', 'x fft bin 6', 'x fft bin 7', 'x fft bin 8', 'x fft bin 9', 'y fft bin 1', 'y fft bin 10', 'y fft bin 2', 'y fft bin 3', 'y fft bin 4', 'y fft bin 5', 'y fft bin 6', 'y fft bin 7', 'y fft bin 8', 'y fft bin 9', 'z fft bin 1', 'z fft bin 10', 'z fft bin 2', 'z fft bin 3', 'z fft bin 4', 'z fft bin 5', 'z fft bin 6', 'z fft bin 7', 'z fft bin 8', 'z fft bin 9']]
            y_test = test['Activity']
            #GridSearch on classifier
            rft = GridSearchCV( classifier,parameters)
            rft.fit(X_train,y_train)
            best_params = rft.best_params_
            best_estimator = rft.best_estimator_
            model_with_best_params = [best_estimator, best_params]
            prediction = rft.predict(X_test)
            predictionprobability = rft.predict_proba(X_test)
            Probability = Probability + list(predictionprobability)
            print("done")
            ###Creating the Emission Probability
        means = (HiddenMarkovModel.mean(Probability,y))
        with open(pklname, 'wb') as f:
           pickle.dump(means, f)
        print("Emission probability generated")
            
            
        
    def HMMapplication(filename,participant,classifier,parameters, classifiername,pklfile,place):
        data = pd.read_csv(filename)
        data.loc[data.Activity == 101, 'Activity'] = 888
        data.loc[data.Activity == 103, 'Activity'] = 888
        data.loc[data.Activity == 104, 'Activity'] = 888
        data = data[data.Activity.isin([888,303])]

        
        
        #Data used for HMM Features
        y =  data['Activity'].loc[(data.Participant.isin(participant))].tolist()
        
        
        Truelabel = []
        Classifier= []
        HiddenMarkov = []
        Probability = []
        
        with open(pklfile, 'rb') as f: #mean(Probability,y) --> saved into a pickle file 
            emmision = pickle.load(f)
        
        predictionperparticipant = []
        for partic in participant:
            dataset =  data.loc[(data.Participant.isin(participant))]
            test = dataset.loc[(data.Participant == partic)]
            train = dataset.loc[(data.Participant != partic)]
            X_train = train[['max', 'min', 'std', 'skew', 'kurtosis', 'mean', 'median', 'x max', 'y max', 'z max', 'x min', 'y min', 'z min', 'x mean', 'y mean', 'z mean', 'x median', 'y median', 'z median', 'x skew', 'y skew', 'z skew', 'x kurtosis', 'y kurtosis', 'z kurtosis', 'x std', 'y std', 'z std', 'xy mean', 'yz mean', 'xz mean','x fft min', 'y fft min', 'z fft min', 'x fft max', 'y fft max', 'z fft max', 'x fft std', 'y fft std', 'z fft std', 'x fft mean', 'y fft mean', 'z fft mean', 'x fft median', 'y fft median', 'z fft median', 'x fft mean weighted', 'y fft mean weighted', 'z fft mean weighted', 'x fft bin 1', 'x fft bin 10', 'x fft bin 2', 'x fft bin 3', 'x fft bin 4', 'x fft bin 5', 'x fft bin 6', 'x fft bin 7', 'x fft bin 8', 'x fft bin 9', 'y fft bin 1', 'y fft bin 10', 'y fft bin 2', 'y fft bin 3', 'y fft bin 4', 'y fft bin 5', 'y fft bin 6', 'y fft bin 7', 'y fft bin 8', 'y fft bin 9', 'z fft bin 1', 'z fft bin 10', 'z fft bin 2', 'z fft bin 3', 'z fft bin 4', 'z fft bin 5', 'z fft bin 6', 'z fft bin 7', 'z fft bin 8', 'z fft bin 9']]
            y_train = train['Activity']
            X_test = (test[['max', 'min', 'std', 'skew', 'kurtosis', 'mean', 'median', 'x max', 'y max', 'z max', 'x min', 'y min', 'z min', 'x mean', 'y mean', 'z mean', 'x median', 'y median', 'z median', 'x skew', 'y skew', 'z skew', 'x kurtosis', 'y kurtosis', 'z kurtosis', 'x std', 'y std', 'z std', 'xy mean', 'yz mean', 'xz mean','x fft min', 'y fft min', 'z fft min', 'x fft max', 'y fft max', 'z fft max', 'x fft std', 'y fft std', 'z fft std', 'x fft mean', 'y fft mean', 'z fft mean', 'x fft median', 'y fft median', 'z fft median', 'x fft mean weighted', 'y fft mean weighted', 'z fft mean weighted', 'x fft bin 1', 'x fft bin 10', 'x fft bin 2', 'x fft bin 3', 'x fft bin 4', 'x fft bin 5', 'x fft bin 6', 'x fft bin 7', 'x fft bin 8', 'x fft bin 9', 'y fft bin 1', 'y fft bin 10', 'y fft bin 2', 'y fft bin 3', 'y fft bin 4', 'y fft bin 5', 'y fft bin 6', 'y fft bin 7', 'y fft bin 8', 'y fft bin 9', 'z fft bin 1', 'z fft bin 10', 'z fft bin 2', 'z fft bin 3', 'z fft bin 4', 'z fft bin 5', 'z fft bin 6', 'z fft bin 7', 'z fft bin 8', 'z fft bin 9']])
            y_test = test['Activity']
            #GridSearch on Random Forest
            rft = GridSearchCV(classifier,parameters)
            rft.fit(X_train,y_train)
            best_params = rft.best_params_
            best_estimator = rft.best_estimator_
            model_with_best_params = [best_estimator, best_params]
            prediction = rft.predict(X_test)
            predictionprobability = rft.predict_proba(X_test)
            #print(predictionprobability)
            predictprobtrain=rft.predict_proba(X_train)
            
            classifierscore = rft.score(X_test,y_test)
            Truelabel = Truelabel + list(y_test)
            Classifier= Classifier + list(prediction)
            Probability = Probability + list(predictionprobability)
            #HMM Part applied per subject
            initialprobabilities = HiddenMarkovModel.initialprobability(y)
            transitionprobabilities =HiddenMarkovModel.transitionprobability(y)
            for i in transitionprobabilities:
                if (sum(i)!=1.0):
                   m=i[7]+(1.0-(sum(i)))
                   i[7]=m
                   
            #means = np.asarray(mean(predictprobtrain,list(y_train)))
            #means = np.asarray(mean(predictionprobability,list(y_test)))
            means = np.asarray(emmision)
            startprob = np.asarray(initialprobabilities)
            #covariance = empirical_covariance(means)
        
            Hmm =  hmm.GaussianHMM(n_components=8,covariance_type="diag")
            
            
            Hmm.fit(predictprobtrain)
            Hmm.startprob_ = startprob
            Hmm.transmat_ = np.asarray(transitionprobabilities)
            Hmm.means_ = means
            Hmm.covars_ =  np.asarray(0.01*np.ones((2,2), dtype=float))
            #Hmm.covars_ = covariance
            Hpredict = Hmm.predict(np.asarray(predictionprobability))
            HiddenMarkov = HiddenMarkov + list(Hpredict)
            
            #print out everything for hmm
            #print("Participant Number",partic)
            print("Classifier score", classifierscore)
            HMMscore = HiddenMarkovModel.Hmmscore(y_test,prediction,Hpredict)
            predictionperparticipant.append([classifierscore,HMMscore])
            
            #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            #    print("Initial probability: \n", pd.DataFrame(Hmm.startprob_,index=["Run/Walk","Crawl","Climb","Stand","Sit","Lie down","Carried","Stroller"])) 
            #    print("Transition probability: \n", pd.DataFrame( Hmm.transmat_,index=["Run/Walk","Crawl","Climb","Stand","Sit","Lie down","Carried","Stroller"],columns = ["Run/Walk","Crawl","Climb","Stand","Sit","Lie down","Carried","Stroller"]))
            #    print("Emission probability: \n",pd.DataFrame(Hmm.means_ ,index=["Run/Walk","Crawl","Climb","Stand","Sit","Lie down","Carried","Stroller"]).T)
        
        
        HiddenMarkovModel.confusionmatrix(Truelabel,Classifier,HiddenMarkov,classifiername,place)

#Classifiers
    def runHMM(datatype,second,emmision,participant):
        models=[]
        
            
        models.append([RandomForestClassifier(), {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400]}, "Random Forest",(datatype+"\EmissionProbability\RandomForestActivevsCarried"+second+".pkl"),""])
        
        
        
        for i in models:
            if emmision == True:
                HiddenMarkovModel.emissionProbability('Feature Extraction Results/Features table - No Timeshift- '+ datatype +' '+ second +' sec .csv',participant,i[0],i[1],i[3])
            HiddenMarkovModel.HMMapplication('Feature Extraction Results/Features table - No Timeshift- '+ datatype +' '+ second +' sec .csv',participant,i[0],i[1],i[2],i[3],i[4])
            

#Waist     
HiddenMarkovModel.runHMM("Waist","5",True,[101,102,103,104,106,107,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124])



