# -*- coding: utf-8 -*-
#Author: aMagicNeko
#Date: 2020.11.02
#Email: smz129@outlook.com
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time 
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class Perceptron(object):
    
    def __init__(self):
        self.learning_step=0.000001
        self.max_iteration=500000
        
    def __predict(self,x):
        x=np.array(x)
        n=len(x)
        x=np.append(x,1)
        return int(np.inner(x,self.w)>0)
        
    def predict(self,features):
        labels=np.zeros(len(features))
        k=0
        for x in features:
            labels[k]=self.__predict(x)
            k+=1
        return labels
        
    def train(self,features,labels):
        correct_count=0
        n=len(labels)
        features=np.array(features)
        shape=list(features.shape)
        shape[-1]=1
        features=np.append(features,np.ones(tuple(shape)),axis=-1)
        self.w=np.zeros(len(features[0]))
        while True:
            k=np.random.randint(0,n)
            if(np.inner(self.w,features[k])*labels[k] > 0):
                correct_count+=1
            else:
                self.w+=self.learning_step * features[k] * labels[k]
            if correct_count>=self.max_iteration:
                break
        return self.w
    
    
if __name__=="__main__":
    print("Start to read datas")
    time0=time.time()
    raw_data=pd.read_csv("train_binary.csv",header=0)
    data=raw_data.values
    imgs=data[:,1:]
    labels=data[:,0]
    train_features,test_features,train_labels,test_labels=\
        train_test_split(imgs,labels,test_size=0.33)
    print(train_features.shape)
    time1=time.time()
    print("Reading data costs ",time1-time0,'seconds\n')
    print("Start training")
    model=Perceptron()
    model.train(train_features,train_labels)
    time2=time.time()
    print("Training costs ",time2-time1,"seconds\n")
    print("Start predict")
    predict_labels=model.predict(test_features)
    time3=time.time()
    print("Predicting costs ",time3-time2,"seconds\n")
    score=accuracy_score(test_labels,predict_labels)
    print("The accuracy score is ",score)