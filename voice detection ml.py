#!/usr/bin/env python
# coding: utf-8

# ### Dans l'optique d'idenfier les voies(masculine,feminine), il nous est demandé de construire un modèle machine learning afin de résoudre ce problème spécifique. Dans ce projet, nous allons utiliser la programmation orienté objet afin de definir l'architeture de notre travail. 

# In[1]:


class FatihamVoiceDetector():
    def __init__(self,model,pd,spliting):
        self.model = model
        self.pd = pd
        self.spliting = spliting
    def load_data(self,directory):
        self.data = self.pd.read_csv(directory)
        
    def visualise_data(self):
        print("------------------Data visualisation------------------")
        self.data.hist(figsize=(20,14))
        
    def statistics(self,transpose = True ):
        print("------------------Statistics--------------------------\n")
        if transpose:
            return self.data.describe().T
        else:
            return self.data.describe()
        
            
    def spliting_data(self,test_size=0.3,random_state=42):
        self.data.iloc[:,-1] = self.data.iloc[:,-1].map({"male":1,"female":0})
        x , y = self.data.iloc[:,:-1], self.data.iloc[:,-1]
        self.xtrain,self.xtest,self.ytrain,self.ytest = self.spliting.train_test_split(x,y,test_size=test_size,random_state=random_state)
    
    def model_train(self):
        print("Training...")
        try:
            self.model = self.model.fit(self.xtrain,self.ytrain)
            print("The model trained...")
        except:
            print("Error during training...")
        
        
    def make_prediction(self):
        print("Predicting....")
        self.predicted = self.model.predict(self.xtest) 
        print("Predicted successful...")
        
    def evaluate(self):
        print("------------------Evaluation----------------\n")
        good = (self.predicted == self.ytest).sum()
        bad = (self.predicted != self.ytest).sum()
        percentage = good/len(self.ytest)
        print(f"Good predition : {good} out of {len(self.ytest)}")
        print(f"Bad predition : {bad} out of {len(self.ytest)}")
        print(f"Accuray :  {round(percentage,2)*100}%")
        
    def custom_prediction(self,x):
        try:
            predicted = self.model.predict(x)
            if predicted == 1:
                print("The voice is male voice")
            else:
                print("The voice is female voice")
        except:
            print("Make sure the size of x match ")
       


# In[2]:


import numpy as np
import pandas as pd
from sklearn import model_selection as splitting
from sklearn.linear_model import LogisticRegression


# In[3]:


model = LogisticRegression()


# In[4]:


fatiham_voice_detect = FatihamVoiceDetector(model=model,spliting=splitting,pd=pd)


# In[5]:


directory = "datasets/voice.csv"


# In[6]:


fatiham_voice_detect.load_data(directory)


# In[7]:


fatiham_voice_detect.visualise_data()


# In[8]:


fatiham_voice_detect.statistics()


# In[9]:


fatiham_voice_detect.spliting_data()


# In[10]:


fatiham_voice_detect.model_train()


# In[11]:


fatiham_voice_detect.make_prediction()


# In[12]:


fatiham_voice_detect.evaluate()


# In[13]:


male_voice_data = np.array([[0.066008740387572,0.0673100287952527,0.040228734810579,0.0194138670478914,0.0926661901358113,0.0732523230879199,22.4232853628204,634.613854542068,0.892193242265734,0.513723842537073,0,0.066008740387572,0.107936553670454,0.0158259149357072,0.25,0.00901442307692308,0.0078125,0.0546875,0.046875,0.0526315789473684]])


# In[14]:


female_voice_data = np.array([[0.193981463168847,0.0522402344705884,0.205424836601307,0.173169934640523,0.225326797385621,0.052156862745098,1.84647373408591,6.14662307809754,0.910067717390582,0.45504003078403,0.223267973856209,0.193981463168847,0.182574178632965,0.0210249671484888,0.275862068965517,0.291666666666667,0.0078125,2.0703125,2.0625,0.116907713498623]])


# In[15]:


fatiham_voice_detect.custom_prediction(male_voice_data)


# In[16]:


fatiham_voice_detect.custom_prediction(female_voice_data)


# ### Nous y sommes, le modèle a fait une prediction correcte, il a predit la voie feminine alors que c'est effectivement une voie feminine et la voie masculine aussi. L'assurence de prediction est de 90%
