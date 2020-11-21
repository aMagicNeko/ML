#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 19:11:28 2020

@author: aMagicNeko

email: smz129@outlook.com
"""
import numpy as np
from itertools import product
import math
class naive_beyesian(object):
    """
    member :
        prior_probability: dict, key is tuple(label)
        condition_probability: dict, key is like ((1,2),lable)
            key[0][0] is the index of the portion, the second
            is the value of the portion
    """
    def __init__(self):
        pass
    
    def train(self,points,labels):
        """
        Build prior probability of labels and
        condition probability of the portions of points
        with labels occuring.
        
        Time:O(m*n^2) on average
        Space:O(ln*sum(pn[k])) pn[k] is the number of possible
        values of portion k. ln is the number of possible values of labels
        Parameters
        ----------
        points : ndarray shape: (n,m)
            points in data set
        labels : ndarray shape: (n)
            DESCRIPTION.
        
        Returns
        -------
        None.

        """
        self.prior_probability=dict()
        self.condition_probability=dict()
        n=len(labels)
        if len(points.shape )!= 2:
            raise ValueError
        m=points.shape[1]
        for label in labels:
            if label in self.prior_probability:
                self.prior_probability[label]+=1
            else :
                self.prior_probability[label]=1
        for k in range(0,n):
            point=points[k]
            label=labels[k]
            for temp in enumerate(point):
                key=tuple([temp,label])
                if key in self.condition_probability:
                    self.condition_probability[key]+=1
                else:
                    self.condition_probability[key]=1
        for key in self.prior_probability:
            self.prior_probability[key]=math.log(self.prior_probability[key])
        for key in self.condition_probability:
            self.condition_probability[key]=math.log(self.condition_probability[key])-self.prior_probability[key[1]]
        for key in self.prior_probability:
            self.prior_probability[key]-=math.log(n)
    def predict(self,p):
        """
        Predict the label of the point p.
        
        Time: O(ln*m) ln is the number of labels
        
        Parameters
        ----------
        p : ndarray shape:(m,)
            point to predict

        Returns
        -------
        label

        """
        m=len(p)
        label=0
        flag=float("-inf")
        for key in self.prior_probability:
            posterior_probability=self.prior_probability[key]
            for k in range(m):
                key1=tuple([tuple([k,p[k]]),key])
                posterior_probability+=self.condition_probability[key1]
            if posterior_probability>flag:
                flag=posterior_probability
                label=key
        return label
if __name__=="__main__":
    data=[[1,'S'],[1,'M'],[1,'M'],[1,'S'],[1,'S'],[2,'S'],[2,'M'],[2,'M'],[2,'L'],[2,'L'],[3,'L'],[3,'M'],[3,'M'],[3,'L'],[3,'L']]
    data=np.array(data)
    labels=[-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]
    model=naive_beyesian()
    model.train(data,labels)
    p=np.array([3,'M'])
    l=model.predict(p)
    for key in model.prior_probability:
        print(key,' ',math.exp(model.prior_probability[key]))
    for key in model.condition_probability:
        print(key,' ',math.exp(model.condition_probability[key]))