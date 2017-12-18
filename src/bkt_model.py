# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:21:30 2017

@author: ayush
"""

import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
from pandas.compat import StringIO
from collections import defaultdict
 

"""
PARAMETERS BELOW -- change here if required
"""
bkt_threshold = 0.99

#BKT parameters
G=0.3
S=0.1
T=0.2
L=0.2


df = pd.read_csv("original_data.csv", delimiter=',',encoding='latin-1')
print(len(df))
#Dropping the 60000 skill_id with NaN
df=df.dropna(subset=['skill_id'])

#Setting NaN in original opportunities to zero 
df['opportunity_original'].fillna(0, inplace=True)


df['overlap_time']=df['overlap_time'].astype(int)
df['answer_type']=df['answer_type'].astype('category')

df['answer_type'] = df['answer_type'].map({'algebra':0,'fill_in_1':1,'choose_1':2,'open_response':3,'choose_n':4})


#Removing unneeded columns 
#del df['order_id']
del df['assignment_id']
del df['assistment_id']
del df['problem_id']
del df['sequence_id']
del df['student_class_id']
del df['position']
del df['type'] # All are mastery
del df['school_id']
del df['template_id']
del df['answer_id']
del df['answer_text']
del df['skill_name']
del df['teacher_id']
#now 338001
#remove scaffolding
df=df.query('original != 0')
# now 312232
del df['original']

df.sort_values(['user_id', 'skill_id','order_id'], ascending=[True, True, True],inplace=True)

#Number of skills 123
len(df["skill_id"].unique())

#Number of students 4163
len(df["user_id"].unique())

#reduces to 18377 pairs if taking more than 10 attempts
g=df.groupby(['user_id', 'skill_id'])
df=g.filter(lambda x: len(x) > 10)
# now has 182690 rows


#Reset The Index
df = df.reset_index(drop=True)

#skill columns
skill = df['skill_id'].unique()
students = df['user_id'].unique()
dict={}
for student in students:
  dict[student]={}

  for sk in skill:
    dict[student][sk]=L
  

df=pd.concat([df,pd.DataFrame(columns=["Prob"])])




#Calculate Probability of KCs
prevStu=""
prevskill=""
for index, row in df.iterrows():
  print(index)

  #First instance of the combo
  if (row["user_id"] != prevStu or row["skill_id"] != prevskill):
    i=0
    prevStu = row["user_id"]
    prevskill = row["skill_id"]
    #print(row)
    # do init for  all student-skills
    L0= dict[row["user_id"]][row["skill_id"]] 

    if (row["correct"]):  
      PC = (L0 * (1-S))/(L0*(1-S)+(1-L0)*G)
      New = PC + (1-PC)*T
      df.ix[index, "Prob"]=New
      dict[row["user_id"]][row["skill_id"]]=New
      if  (New > 1):
        df.ix[index, "Prob"]=1
        dict[row["user_id"]][row["skill_id"]]=1

    #Incorrect
    else:  
      PC = (L0 *S)/(L0*S+(1-L0)*(1-G))
      New = PC + (1-PC)*T
      df.ix[index, "Prob"]=New
      dict[row["user_id"]][row["skill_id"]]=New
      if  (New > 1):
        df.ix[index, "Prob"]=1
        dict[row["user_id"]][row["skill_id"]]=1
          
  # Not the first instance of the combo
  else:
    
    # No need to calculate after the first 20 rows
    if(i>20):
      continue
    i+=1
    prevStu = row["user_id"]


    prevL = dict[row["user_id"]][row["skill_id"]]#df.ix[index-1, pCols[i]]

    if (row["correct"]):

      PC = (prevL * (1-S))/(prevL*(1-S)+(1-prevL)*G)
      New = PC + (1-PC)*T
      df.ix[index, "Prob"]=New
      dict[row["user_id"]][row["skill_id"]]=New
      if  (New > 1):
          df.ix[index, "Prob"]=1
          dict[row["user_id"]][row["skill_id"]]=1
    
    #Incorrect
    else:

      PC = (prevL *S)/(prevL*S+(1-prevL)*(1-G))
      New = PC + (1-PC)*T
      df.ix[index, "Prob"]=New
      dict[row["user_id"]][row["skill_id"]]=New
      if  (New > 1):
        df.ix[index, "Prob"]=1
        dict[row["user_id"]][row["skill_id"]]=1



#Map the BKT values to 0 or 1 based on threshold
df['bin'] = ((df.Prob) > bkt_threshold)
df['bin'] = df['bin'].map({False : 1, True : 0})


df.to_csv("cleaned.csv")



















