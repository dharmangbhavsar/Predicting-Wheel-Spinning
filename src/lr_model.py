# -*- coding: utf-8 -*-
"""
Created on Fri Nov  21 07:36:53 2017

@author: Dharmang
"""

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
le = preprocessing.LabelEncoder()
le = preprocessing.LabelEncoder()

df = pd.read_csv("cleaned.csv", delimiter=',',encoding='latin-1')


result = pd.DataFrame(columns=['A','B','C','D','E','F','G','H','I'])
result1 = pd.DataFrame(columns=['A','B','C','D','E','F','G','H','I'])
result2 = pd.DataFrame(columns=['A','B','C','D','E','F','G','H','I'])
result3 = pd.DataFrame(columns=['A','B','C','D','E','F','G','H','I'])

#Encode Tutor Mode as nominal value
df['tutor_mode'] = le.fit_transform(df['tutor_mode'])
df['bottom_hint'].fillna(0, inplace=True)
#deleting unneeded column
del df['Unnamed: 0']


#creting a key column for student skill
df['key']= df['user_id'].map(str) + '-' + df['skill_id'].map(str)
del df['user_id']
del df['skill_id']
del df['opportunity_original'] #  it has high correlation to opportunity
df['key']=df['key'].astype(str)
df=df.rename(columns={'bin':'bindf'})
#deciding value of n
n=5
cm = [[]]*5
folds=5
skf = StratifiedKFold(n_splits=folds,random_state=0)
for n in range(5,11):
  print("--------n="+str(n)+"---------------------")
#getting the nth row for each key and just keeping the key and bin columns
  g=df.groupby(['key']).nth(n-1) # as it is zero indexed
  g['key'] = g.index
  lists=['key','bindf']
  g=g[lists]
  g=g.rename(columns={'bindf':'bing'})
  
  
  #Create the n folds
  skf.get_n_splits(g['key'], g['bing'])
  j=-1
  final_row=[]
  final_row1=[]
  final_row2=[]
  final_row3=[]
  X=g['key']
  y=g['bing']
  
  #folds
  for train_index, test_index in skf.split(g['key'], g['bing']):
    j+=1
    print("Fold "+str(j))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]  
    
  #X_train, X_test, y_train, y_test = train_test_split(g['key'], g['bing'], test_size=0.33, random_state=0)
    X_train=pd.DataFrame(X_train)
    X_test=pd.DataFrame(X_test)
    y_train=pd.DataFrame(y_train)
    y_test=pd.DataFrame(y_test)
    y_train=y_train.sort_index()
    y_test=y_test.sort_index()
    
    #subset the bigger dataframe with only keys present for this particular n value
    df_train=df.merge(X_train,left_on='key', right_on='key', how='inner')
    
    del df_train['bindf']
    del df_train['Prob']
    del df_train['order_id']
    
    df_test=df.merge(X_test,left_on='key', right_on='key', how='inner')
    
    del df_test['bindf']
    del df_test['Prob']
    del df_test['order_id']
    
    row=[]
    row1=[]
    row2=[]
    row3=[]
    X_train=df_train.groupby(['key']).nth(0)
    X_test=df_test.groupby(['key']).nth(0)
    
    # create models for upto the nth attempt
    for i in range(1,n):
      X_traini=df_train.groupby(['key']).nth(i)
      X_testi=df_test.groupby(['key']).nth(i)
      if (i>0):
        
        # Merge the latest attempt to the dataset for the next iteration of the model
        X_train=pd.merge(X_train, X_traini, left_index=True, right_index=True)
        X_test=pd.merge(X_test, X_testi, left_index=True, right_index=True)
      else:
        X_train=X_traini
        X_test= X_testi
     
      logit1 = LinearSVC(C=1.0)
      #logit1 = LogisticRegression(C=1.0)
      logit1.fit(X_train,y_train)
      
      pred=logit1.predict(X_test)
      #cf = confusion_matrix(y_test,pred)
      tn, fp, fn, tp = confusion_matrix(y_test,pred).ravel()
      
      #accuracy
      a=(tp+tn)/(tp+tn+fp+fn)

      #precision
      b=tp/(tp+fp)

      #recall
      d=tp/(tp+fn)
      
      #f1score
      c=2*(b*d)/(b+d)
      
      #Append the values generated for this attempt in the array
      row = row+[a]
      row1+=[b]
      row2+=[c]
      row3+=[d]
    
    
    #For this fold, add the respective measurement values to the final_row array
    if len(final_row)==0:
      final_row=row
    else:
      final_row=[x + y for x, y in zip(row, final_row)]
     
    if len(final_row1)==0:
      final_row1=row1
    else:
      final_row1=[x + y for x, y in zip(row1, final_row1)]
      
    if len(final_row2)==0:
      final_row2=row2
    else:
      final_row2=[x + y for x, y in zip(row2, final_row2)]
      
    if len(final_row3)==0:
      final_row3=row3
    else:
      final_row3=[x + y for x, y in zip(row3, final_row3)]
      
  
  # Get average of values across folds  
  final_row=[i/folds for i in final_row]  
  final_row1=[i/folds for i in final_row1]  
  final_row2=[i/folds for i in final_row2]  
  final_row3=[i/folds for i in final_row3]
  cm.append(confusion_matrix(y_test,pred))  
  
  #Fill NA in the empty cells of the final dataframe
  diff=len(result.columns) - len(final_row)
  final_row = final_row + [None]*diff
  result.loc[n] = final_row
            
  diff1=len(result1.columns) - len(final_row1)
  final_row1 = final_row1 + [None]*diff1
  result1.loc[n] = final_row1
             
  diff2=len(result2.columns) - len(final_row2)
  final_row2 = final_row2 + [None]*diff2
  result2.loc[n] = final_row2
             
  diff3=len(result3.columns) - len(final_row3)
  final_row3 = final_row3 + [None]*diff3
  result3.loc[n] = final_row3
 
"""
# Code to print the dataframes of the measurements across different n values and attempts
print(result)  
print(result1)
print(result2)
print(result3)
"""

#From here we print the metric scores for the prediction using the model fitted earlier

#This is the graph for Accuracy vs number of attempts
#Attempts from 5 to 10 are plotted for its accuracy
#The scores are in the form of a dataframe saved in result
for i in range(0,6):
    print(i)
    if i==5:
        y = result.iloc[5,:]
        x = np.arange(1,10)
        plt.plot(x,y)
    else:
        y = result.iloc[i,0:(5+i)]
        x = np.arange(1,6+i)
        plt.plot(x,y)
plt.legend(['n=5','n=6','n=7','n=8','n=9','n=10'],loc = 'top right')
plt.xlabel('number of attempts')
plt.ylabel('Accuracy score')
plt.suptitle('Accuracy score vs Number of attempts')
plt.savefig("Accuracy.png")
plt.show() 
   
#This is the graph for Precision vs number of attempts
#Attempts from 5 to 10 are plotted for its precision
#The scores are in the form of a dataframe saved in result1
for i in range(0,6):
    print(i)
    if i==5:
        y = result1.iloc[5,:]
        x = np.arange(1,10)
        plt.plot(x,y)
    else:
        y = result1.iloc[i,0:(5+i)]
        x = np.arange(1,6+i)
        plt.plot(x,y)
plt.legend(['n=5','n=6','n=7','n=8','n=9','n=10'],loc = 'top right')
plt.xlabel('number of attempts')
plt.ylabel('Precision score')
plt.suptitle('Precision score vs Number of attempts')
plt.savefig("Precision.png")
plt.show()  

#This is the graph for F1 score vs number of attempts
#Attempts from 5 to 10 are plotted for its F1 score
#The scores are in the form of a dataframe saved in result2
for i in range(0,6):

    if i==5:
        y = result2.iloc[5,:]
        x = np.arange(1,10)
        plt.plot(x,y)
    else:
        y = result2.iloc[i,0:(5+i)]
        x = np.arange(1,6+i)
        plt.plot(x,y)
plt.legend(['n=5','n=6','n=7','n=8','n=9','n=10'],loc = 'top right')
plt.xlabel('number of attempts')
plt.ylabel('F1 score score')
plt.suptitle('F1 score score vs Number of attempts')
plt.savefig("F1.png")
plt.show() 

#This is the graph for Recall vs number of attempts
#Attempts from 5 to 10 are plotted for its Recall score
#The scores are in the form of a dataframe saved in result3
for i in range(0,6):
    print(i)
    if i==5:
        y = result3.iloc[5,:]
        x = np.arange(1,10)
        plt.plot(x,y)
    else:
        y = result3.iloc[i,0:(5+i)]
        x = np.arange(1,6+i)
        plt.plot(x,y)
plt.legend(['n=5','n=6','n=7','n=8','n=9','n=10'],loc = 'top right')
plt.xlabel('number of attempts')
plt.ylabel('Recall score')
plt.suptitle('Recall score vs Number of attempts')
plt.savefig("Recall.png")
plt.show()