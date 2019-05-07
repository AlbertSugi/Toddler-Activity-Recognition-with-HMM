# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:22:04 2019

@author: Albert
"""

import pandas as pd
import itertools
import numpy as np
import math
import scipy.stats.stats as stats
import matplotlib.pyplot as plt 
import pickle
#
#
#
#Converts 
#"Raw Data/Annotation_Sheet(P108&P102&P113 - Edit).xlsx"
#"Raw Data/WristWaistFile/*waist*.csv"
#'Features table - No Timeshift- Wrist .csv'
class Conversion():
    def mergenextract(filename,rawsignal,outputfilename,clip):
        xls = pd.ExcelFile(filename)
        
        dfs = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(filename, sheetname=sheet_name)
            dfs.append(df)
            
        annotation = pd.concat(dfs, join='inner')
        
        
        # In[87]:
        
        
        #Adding the Start time and end time of the clips as new names with date
        
        annotation = annotation[annotation["Date"].notnull()]
        annotation = annotation[annotation["StartClock"].notnull()]
        annotation = annotation[annotation["EndClock"].notnull()]
        
        annotation['DateStartTimeClock'] = pd.to_datetime(annotation['Date'].apply(str)+' '+annotation['StartClock'].apply(str))
        annotation['DateEndTimeClock'] = pd.to_datetime(annotation['Date'].apply(str)+' '+annotation['EndClock'].apply(str))
#        def participanttimeshift(participantid,timeshift):
#            annotation['DateStartTimeClock'].loc[annotation['ParticipantID']==participantid]=annotation['DateStartTimeClock'].loc[annotation['ParticipantID']==participantid]+pd.Timedelta(seconds=timeshift)
#            annotation['DateEndTimeClock'].loc[annotation['ParticipantID']==participantid]=annotation['DateEndTimeClock'].loc[annotation['ParticipantID']==participantid]+pd.Timedelta(seconds=timeshift)
#        
            #participanttimeshift(101,-1)
            #participanttimeshift(102,-3)
            #participanttimeshift(103,41)
            #participanttimeshift(104,2)
            #participanttimeshift(106,2)
            #participanttimeshift(107,1)
            #participanttimeshift(110,24)
            #participanttimeshift(111,33)
            #participanttimeshift(112,3)
            #participanttimeshift(113,-1)
            #participanttimeshift(114,92)
            #participanttimeshift(115,1)
            #participanttimeshift(117,-76)
            #participanttimeshift(121,1)
            #participanttimeshift(122,-17)
            #participanttimeshift(124,71)
        
        
        print(annotation.shape)
        annotation.head()
        print(annotation.dtypes)
        
        
        
        
        
        annotation[annotation["ActivityLevel1"]==999].count()
        
        
        
        
        
        annotation[annotation["ActivityLevel2"]==999].count()
        
        
        
        
        
        annotation[annotation["ActivityLevel3"]==999].count()
        
        
        
        
        annotation = annotation[annotation["ActivityLevel1"]!=999] #1
        
        
        
        
        
        annotation.shape
        
        
        
        
        
        annotation[annotation["ActivityLevel1"]==999].count()
        
        
        # In[94]:
        
        
        annotation[annotation["ActivityLevel1"].isnull()].count()
        
        
        # In[95]:
        
        
        annotation[annotation["ActivityLevel2"].isnull()].count()
        
        
        # In[96]:
        
        
        annotation[annotation["ActivityLevel3"].isnull()].count()
        
        
        # In[97]:
        
        
        annotation = annotation[annotation["ActivityLevel1"].notnull()]#2
        annotation = annotation[annotation["ActivityLevel2"].notnull()]
        annotation = annotation[annotation["ActivityLevel3"].notnull()]
        
        
        
        
        
        annotation.shape
        
        
        
        
        annotation[annotation["ActivityLevel1"].isnull()].count()
        
        
        
        
        
        annotation[annotation["ActivityLevel3"].isnull()].count()
        
        
        
        
        
        annotation[annotation["ActivityLevel2"].isnull()].count()
        
        
        
        
        
        annotation.ActivityLevel1.unique()#3
        
        
        # In[103]:
        
        
        annotation.ActivityLevel2.unique()
        
        
        # In[104]:
        
        
        annotation.ActivityLevel3.unique()
        
        
        # In[105]:
        
        
        annotation.loc[annotation.ActivityLevel1 == 3, 'ActivityLevel1'] = 2 #4
        
        
        # In[106]:
        
        
        annotation.ActivityLevel1.unique()
        
        
        # In[107]:
        
        
        annotation.loc[annotation.ActivityLevel2 == 2, 'ActivityLevel2'] = 20
        
        
        # In[108]:
        
        
        annotation.ActivityLevel2.unique()
        
        
        # In[109]:
        
        
        annotation.loc[annotation.ActivityLevel3 == 207, 'ActivityLevel3'] = 307
        
        
        # In[110]:
        
        
        annotation.ActivityLevel3.unique()
        
        
        # In[111]:
        
        
        testdf = annotation.loc[annotation['ActivityLevel1'] == 1]
        testdf.groupby(['ActivityLevel2'])['ActivityLevel2'].count() #5
        
        
        # In[112]:
        
        
        annotation.loc[((annotation.ActivityLevel1 == 1) & (annotation.ActivityLevel2 == 30)), 'ActivityLevel1'] = 2 #6
        
        
        # In[113]:
        
        
        testdf = annotation.loc[annotation['ActivityLevel1'] == 1]
        testdf.groupby(['ActivityLevel2'])['ActivityLevel2'].count()
        
        
        # In[114]:
        
        
        testdf = annotation.loc[annotation['ActivityLevel1'] == 2]     #7
        testdf.groupby(['ActivityLevel2'])['ActivityLevel2'].count()
        
        
        # In[115]:
        
        
        annotation.loc[((annotation.ActivityLevel1 == 2) & (annotation.ActivityLevel2 == 10)), 'ActivityLevel1'] = 1 #8
        
        
        # In[116]:
        
        
        testdf = annotation.loc[annotation['ActivityLevel1'] == 2]
        testdf.groupby(['ActivityLevel2'])['ActivityLevel2'].count()
        
        
        # In[117]:
        
        
        annotation[(annotation.ActivityLevel1 == 2) & (annotation.ActivityLevel2 == 20) & (annotation.ActivityLevel3.isin([201,202,203,204,205,206]))].count() #9
        
        
        # In[118]:
        
        
        annotation.loc[((annotation.ActivityLevel1 == 2) & (annotation.ActivityLevel2 == 20) & (annotation.ActivityLevel3.isin([201,202,203,204,205,206]))), 'ActivityLevel1'] = 1
        
        
        # In[119]:
        
        
        annotation[(annotation.ActivityLevel1 == 2) & (annotation.ActivityLevel2 == 20) & (annotation.ActivityLevel3.isin([201,202,203,204,205,206]))].count() #10
        
        
        # In[120]:
        
        
        testdf = annotation.loc[annotation['ActivityLevel1'] == 2] 
        testdf.groupby(['ActivityLevel2'])['ActivityLevel2'].count()
        
        
        # In[121]:
        
        
        annotation.loc[((annotation.ActivityLevel1 == 2) & (annotation.ActivityLevel2 == 20) & (annotation.ActivityLevel3.isin([307]))), 'ActivityLevel2'] = 30 #11
        
        
        # In[122]:
        
        
        testdf = annotation.loc[annotation['ActivityLevel1'] == 2]
        testdf.groupby(['ActivityLevel2'])['ActivityLevel2'].count()
        
        
        # In[123]:
        
        
        annotation = annotation.drop(annotation[((annotation.ActivityLevel1 == 2) & (annotation.ActivityLevel2 == 20) & (annotation.ActivityLevel3.isin([301])))].index)
        
        
        # In[124]:
        
        
        testdf = annotation.loc[annotation['ActivityLevel1'] == 2]
        testdf.groupby(['ActivityLevel2'])['ActivityLevel2'].count()
        
        
        # In[125]:
        
        
        testdf = annotation.loc[annotation['ActivityLevel2'] == 10]
        testdf.groupby(['ActivityLevel3'])['ActivityLevel3'].count()
        
        
        # In[126]:
        
        
        annotation = annotation.drop(annotation[((annotation.ActivityLevel2 == 10) & (annotation.ActivityLevel1 == 1) & (annotation.ActivityLevel3.isin([201,202,203,204,205,206,301,302,303,304,305,306,307])))].index)
        
        
        # In[127]:
        
        
        testdf = annotation.loc[annotation['ActivityLevel2'] == 10]
        testdf.groupby(['ActivityLevel3'])['ActivityLevel3'].count()
        
        
        # In[128]:
        
        
        testdf = annotation.loc[annotation['ActivityLevel2'] == 20]
        testdf.groupby(['ActivityLevel3'])['ActivityLevel3'].count()
        
        
        # In[129]:
        
        
        annotation = annotation.drop(annotation[((annotation.ActivityLevel1 == 1) & (annotation.ActivityLevel2 == 20) & (annotation.ActivityLevel3.isin([101,102,103,104,105,106,107,301,302,303,304,305,306,307])))].index)
        
        
        # In[130]:
        
        
        testdf = annotation.loc[annotation['ActivityLevel2'] == 20]
        testdf.groupby(['ActivityLevel3'])['ActivityLevel3'].count()
        
        
        # In[131]:
        
        
        testdf = annotation.loc[annotation['ActivityLevel2'] == 30]
        testdf.groupby(['ActivityLevel3'])['ActivityLevel3'].count()
        
        
        # In[132]:
        
        
        annotation.head()
        
        
        # In[133]:
        
        
        annotation.shape
        
        
        # In[134]:
        
        
        annotation.ActivityLevel3.unique()
        
        
        # In[135]:
        
        
        annotation.loc[(annotation.ActivityLevel3.isin([302,303,304,301,201,101,102,103,104,105]))]
        
        
        # In[136]:
        
        
        annotation.shape
        
        
        # In[137]:
        
        
        annotation = annotation.loc[(annotation.ActivityLevel3.isin([302,303,304,301,201,101,102,103,104,105]))]
        
        
        # In[138]:
        
        
        annotation.shape
        
        
        # In[139]:
        
        
        annotation.ActivityLevel3.unique()
        
        
        # In[140]:
        
        
        annotation.loc[(annotation.ActivityLevel3 == 102), 'ActivityLevel3'] = 101
        
        
        # In[141]:
        
        
        annotation.loc[(annotation.ActivityLevel3 == 105), 'ActivityLevel3'] = 104
        
        
        # In[142]:
        
        
        annotation.ActivityLevel3.unique()
        
        
        # In[143]:
        
        
        def fetchSignal(file_, df):
            print(df.shape)
            participant = file_.split("_")[0][-3:]
            print(participant)
            participantannotation = annotation.loc[annotation['ParticipantID'] == float(participant)]
            print(participantannotation.shape)
            print(annotation.shape)
            tempframe = pd.DataFrame()
            for index, row in participantannotation.iterrows():
                signalrange = (df['Timestamp'] >= row['DateStartTimeClock']) & (df['Timestamp'] < row['DateEndTimeClock'])
                signalframe = df.loc[signalrange]
                signalframe['activity1'] = row['ActivityLevel1']
                signalframe['activity2'] = row['ActivityLevel2']
                signalframe['activity3'] = row['ActivityLevel3']
                signalframe['Accel_Waist'] = row['Accel_Waist']
                signalframe['Accel_Wrist'] = row['Accel_Wrist']
                signalframe['Sequence'] = row['Sequence']
                signalframe['ParticipantID'] = row['ParticipantID']
                tempframe = tempframe.append(signalframe)
            df = tempframe
            print(df.shape)
            return df
        
        
        # In[145]:
        
        
        #Read all csv files in the folder and ignore the first 10 lines
        import glob
        allFiles = glob.glob(rawsignal)
        list_ = []
        print(allFiles)
        for file_ in allFiles:
            df = pd.read_csv(file_,index_col=None, header=0,skiprows=10)
            print("Processing file ", file_)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = fetchSignal(file_,df)
            list_.append(df)
            print("Finished Processing file ", file_)
        dataframe = pd.concat(list_)
        
        print("Column headings:")
        print(dataframe.columns)
        print(dataframe.dtypes)
        print(dataframe.shape)
        dataframe.head()
        
        
        # In[146]:
        
        
        dataframe = dataframe.reset_index(drop=True)
        
        
        # In[150]:
        
        
        data = None
        data = dataframe
        x = 0
        i = 0
        j = 0
        array = []
        d = []
        actvalue = ''
        subvalue = ''
        for ind, ro in data.iterrows():
            if ((j < clip) and ((actvalue == data.iloc[ind,6]) or actvalue == '') and ((subvalue == data.iloc[ind,10]) or subvalue == '')) :
                activity1 = data.iloc[ind,4]
                activity2 = data.iloc[ind,5]
                actvalue = data.iloc[ind,6]
                waist = data.iloc[ind,7]
                wrist = data.iloc[ind,8]
                sequence = data.iloc[ind,9]
                subvalue = data.iloc[ind,10]
                array.append([data.iloc[ind,1],data.iloc[ind,2],data.iloc[ind,3]])
                j+=1
            else :
                if(j == clip):
                    d.append([subvalue,actvalue,array])
                array = []
                i+=1
                j = 0
            actvalue = data.iloc[ind,6]
            subvalue = data.iloc[ind,10]
        
        dfal3 = pd.DataFrame(d)
        dfal3.head()
        dfal3[0].unique()
        
        
        # In[148]:
        
        
    #    pickling_on = open("10sec.pickle","wb")
    #    pickle.dump(dfal3, pickling_on)
    #    pickling_on.close()
        
        
        # In[ ]:
        
        #
        dfal3[0].unique()
        
        
        # In[64]:
        
        
        def movingaverage(interval, window_size):
            window= np.ones(int(window_size))/float(window_size)
            return np.convolve(interval, window, 'same')
        
        
        # In[65]:
        
        
        def fft_positive(fft):
            half = math.floor(len(fft)/2)
            return fft[1:half]
        
        
        # In[66]:
        
        
        def weighted_fft_mean(signal):
            fft = abs(np.fft.fft(signal))
            t = np.arange(len(signal))
            fft_freq = np.fft.fftfreq(t.shape[-1])
            
            half = math.floor(len(fft)/2)    
            fft = fft[1:half]
            fft_freq = fft_freq[1:half]
            
            mean = np.sum(fft * fft_freq) / sum(fft_freq)
            
            return mean
        
        
        # In[67]:
        
        
        def fft_bin(axis,signal,bins):   
            bin_size = math.floor(len(signal)/bins)
        
            fft_features = [axis + ' fft bin ' + str(i+1) for i in range(bins)]    
        #     fft_bins_mean = pd.DataFrame(index=[0], columns=fft_features)
            
            fft_bins = [signal[i:i + bin_size] for i in range(0, len(signal), bin_size)]
            fft_bins_mean = [np.mean(fft_bins[i]) for i in range(bins)]
            
        #     for i in range(0,bins):
        #         bin_mean_curr = np.mean(fft_bins[i]) 
        #         fft_bins_mean.set_value(0, fft_features[i], bin_mean_curr)        
            
            return dict(zip(fft_features, fft_bins_mean))
        
        
        # In[68]:
        
        
        dfnew = dfal3
       
        dfnew['x'] = dfnew.apply(lambda row:  [i[0] for i in row[2]], axis = 1)
        dfnew['y'] = dfnew.apply(lambda row:  [i[1] for i in row[2]], axis = 1)
        dfnew['z'] = dfnew.apply(lambda row:  [i[2] for i in row[2]], axis = 1)
        dfnew['magnitude'] = dfnew.apply(lambda row: [math.sqrt(i[0]**2 + i[1]**2 + i[2]**2) for i in row[2]], axis = 1)
        dfnew['max'] = dfnew.apply(lambda row:  max(row['magnitude']), axis = 1)
        dfnew['min'] = dfnew.apply(lambda row:  min(row['magnitude']), axis = 1)
        dfnew['std'] = dfnew.apply(lambda row:  np.std(row['magnitude']), axis = 1)
        dfnew['skew'] = dfnew.apply(lambda row:  stats.skew(row['magnitude']), axis = 1)
        dfnew['kurtosis'] = dfnew.apply(lambda row:  stats.kurtosis(row['magnitude']), axis = 1)
        dfnew['mean'] = dfnew.apply(lambda row:  np.mean(row['magnitude']), axis = 1)
        dfnew['median'] = dfnew.apply(lambda row:  np.median(row['magnitude']), axis = 1)
        dfnew['x max'] = dfnew.apply(lambda row:  max(row['x']), axis = 1)
        dfnew['y max'] = dfnew.apply(lambda row:  max(row['y']), axis = 1)
        dfnew['z max'] = dfnew.apply(lambda row:  max(row['z']), axis = 1)
        dfnew['x min'] = dfnew.apply(lambda row:  min(row['x']), axis = 1)
        dfnew['y min'] = dfnew.apply(lambda row:  min(row['y']), axis = 1)
        dfnew['z min'] = dfnew.apply(lambda row:  min(row['z']), axis = 1)
        dfnew['x mean'] = dfnew.apply(lambda row:  np.mean(row['x']), axis = 1)
        dfnew['y mean'] = dfnew.apply(lambda row:  np.mean(row['y']), axis = 1)
        dfnew['z mean'] = dfnew.apply(lambda row:  np.mean(row['z']), axis = 1)
        dfnew['x median'] = dfnew.apply(lambda row:  np.median(row['x']), axis = 1)
        dfnew['y median'] = dfnew.apply(lambda row:  np.median(row['y']), axis = 1)
        dfnew['z median'] = dfnew.apply(lambda row:  np.median(row['z']), axis = 1)
        dfnew['x skew'] = dfnew.apply(lambda row:  stats.skew(row['x']), axis = 1)
        dfnew['y skew'] = dfnew.apply(lambda row:  stats.skew(row['y']), axis = 1)
        dfnew['z skew'] = dfnew.apply(lambda row:  stats.skew(row['z']), axis = 1)
        dfnew['x kurtosis'] = dfnew.apply(lambda row:  stats.kurtosis(row['x']), axis = 1)
        dfnew['y kurtosis'] = dfnew.apply(lambda row:  stats.kurtosis(row['y']), axis = 1)
        dfnew['z kurtosis'] = dfnew.apply(lambda row:  stats.kurtosis(row['z']), axis = 1)
        dfnew['x std'] = dfnew.apply(lambda row:  np.std(row['x']), axis = 1)
        dfnew['y std'] = dfnew.apply(lambda row:  np.std(row['y']), axis = 1)
        dfnew['z std'] = dfnew.apply(lambda row:  np.std(row['z']), axis = 1)
        dfnew['xy mean'] = dfnew.apply(lambda row:  np.mean(np.array(row['x']) * np.array(row['y'])), axis = 1)
        dfnew['yz mean'] = dfnew.apply(lambda row:  np.mean(np.array(row['y']) * np.array(row['z'])), axis = 1)
        dfnew['xz mean'] = dfnew.apply(lambda row:  np.mean(np.array(row['x']) * np.array(row['z'])), axis = 1)
        temp = []
        tempx = []
        tempy = []
        tempz = []
        xrun = []
        yrun = []
        zrun = []
        for i in range(len(dfnew)):
            temp = dfnew.iloc[i,2]
            
            for j in range(len(temp)):
                tempx.append(temp[j][0])
                tempy.append(temp[j][1])
                tempz.append(temp[j][2])
            x = np.array(tempx)
            y = np.array(tempy)
            z = np.array(tempz)
            
            xrun.append(movingaverage(x,20))
            yrun.append(movingaverage(y,20))
            zrun.append(movingaverage(z,20))
            tempx = []
            tempy = []
            tempz = []
        
        dfnew.insert(7, "running x", xrun)
        dfnew.insert(8, "running y", yrun)
        dfnew.insert(9, "running z", zrun)
        x = dfnew['x'] - dfnew['running x']
        y = dfnew['y'] - dfnew['running y']
        z = dfnew['z'] - dfnew['running z']
        
     
        
        x_fft = dfnew.apply(lambda row: [i for i in fft_positive(abs(np.fft.fft(row['x'])))], axis = 1)
        y_fft = dfnew.apply(lambda row: [i for i in fft_positive(abs(np.fft.fft(row['y'])))], axis = 1)
        z_fft = dfnew.apply(lambda row: [i for i in fft_positive(abs(np.fft.fft(row['z'])))], axis = 1)
        
        dfnew.insert(10, 'x fft', x_fft)
        dfnew.insert(11, 'y fft', y_fft)
        dfnew.insert(12, 'z fft', z_fft)
        dfnew['x fft min'] = dfnew.apply(lambda row: min(row['x fft']), axis = 1)
        dfnew['y fft min'] = dfnew.apply(lambda row: min(row['y fft']), axis = 1)
        dfnew['z fft min'] = dfnew.apply(lambda row: min(row['z fft']), axis = 1)
        dfnew['x fft max'] = dfnew.apply(lambda row: max(row['x fft']), axis = 1)
        dfnew['y fft max'] = dfnew.apply(lambda row: max(row['y fft']), axis = 1)
        dfnew['z fft max'] = dfnew.apply(lambda row: max(row['z fft']), axis = 1)
        dfnew['x fft std'] = dfnew.apply(lambda row: np.std(row['x fft']), axis = 1)
        dfnew['y fft std'] = dfnew.apply(lambda row: np.std(row['y fft']), axis = 1)
        dfnew['z fft std'] = dfnew.apply(lambda row: np.std(row['z fft']), axis = 1)
        dfnew['x fft mean'] = dfnew.apply(lambda row: np.mean(row['x fft']), axis = 1)
        dfnew['y fft mean'] = dfnew.apply(lambda row: np.mean(row['y fft']), axis = 1)
        dfnew['z fft mean'] = dfnew.apply(lambda row: np.mean(row['z fft']), axis = 1)
        dfnew['x fft median'] = dfnew.apply(lambda row: np.median(row['x fft']), axis = 1)
        dfnew['y fft median'] = dfnew.apply(lambda row: np.median(row['y fft']), axis = 1)
        dfnew['z fft median'] = dfnew.apply(lambda row: np.median(row['z fft']), axis = 1)
        dfnew['x fft mean weighted'] = dfnew.apply(lambda row: weighted_fft_mean(row['x']), axis = 1)
        dfnew['y fft mean weighted'] = dfnew.apply(lambda row: weighted_fft_mean(row['y']), axis = 1)
        dfnew['z fft mean weighted'] = dfnew.apply(lambda row: weighted_fft_mean(row['z']), axis = 1)
        dfnew = dfnew.merge(dfnew.x.apply(lambda row: pd.Series(fft_bin('x',row,10))), left_index=True, right_index=True)
        dfnew = dfnew.merge(dfnew.y.apply(lambda row: pd.Series(fft_bin('y',row,10))), left_index=True, right_index=True)
        dfnew = dfnew.merge(dfnew.z.apply(lambda row: pd.Series(fft_bin('z',row,10))), left_index=True, right_index=True)
        dfnew.rename(columns={dfnew.columns[1]:"Participant",dfnew.columns[2]:"Activity"})
        #dfnew.drop(columns=(dfnew.columns[3]),axis=1)
        
        
        
        #
        #
        dfnew.to_csv(outputfilename)


#Conversion.mergenextract("Raw Data/Pilot2017_AnnotationDatabaseID.xlsx","Raw Data/WristWaistFile/*wrist*.csv",'Feature Extraction Results/Features table - No Timeshift- Wrist 10 sec Test .csv',300)
        


