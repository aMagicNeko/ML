# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:38:23 2020

@author: aMagicNeko

email: smz129@outlook.com


"""
import pandas as pd
import numpy as np
import time
def loadData(fileName):
    """
    Load datas from the csv file.
    
    Parameters
    ----------
    fileName : string
    the path and name of the csv file
     
    Returns
    -------
    datasets,labelsets

    """
    print("Start to read file")
    fr=pd.read_csv(fileName,header=0)
    dataArr=fr.iloc[:,1:].values
    labelArr=fr.iloc[:,0].values
    return dataArr,labelArr

def LpDistance(x,y,p=2):
    """
    Compute the Lp Distance between x and y.

    Parameters
    ----------
    x : numpy.array
        the first point
    y : numpy.array
        the second point
    p : int
        decide the Lp space
        p=-1 : Chebyshev distance
    Return
    -------
    distance

    """
    if(p>=2):
        return (np.sum((x-y)**p))**(1/p)
    elif(p==1):
        return np.sum(np.abs(x-y))
    elif(p==-1):
        return max(np.abs(x-y))
    else:
        raise ValueError

class kdTree(object):
    """
    members:
        data : np.array
            datas array
        dn : int
            dimension of datas
        n_datas : int
            number of datas
        p : int
            clarify distance
        area : np.array shape=(n_datas,2)
            define the area of each point,
            use int_index to get,
            the first is start index,
            the last is stop index (not included)
            leave's area is -1,-1
    """
    def __init__(self,data,p=2):
        """
        don't change datas after creating.

        Parameters
        ----------
        data : ndarray shape: (n,m)
            data points 
        p : int, optional
            DESCRIPTION. The default is 2.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.data=data
        if(len(data.shape) != 2 ):
            raise ValueError
        self.dn=data.shape[1] #dimension number
        self.n_datas=data.shape[0] #datas numbers
        self.p=p
        self.area=np.zeros((self.n_datas,3),dtype=int)-1
        
    def construct(self):
        """
        construct the KDtree
        
        Notice different ares represented by different
        leaves may intersect on the dividing line.
        Time complexity: O(nlogn)
        
        Returns
        -------
        None.

        """
        self.__construct(0,0,self.n_datas)
        
    def __construct(self,d,s,f):
        """
        
        Time complexity: O(nlogn)
        
        Parameters
        ----------
        d : int
            depth at present
        s : int
            start index
        f : int 
            stop index (not included)
        Returns
        -------
        None

        """
        if(s==f-1 or s==f):
            return #if there's only one data
        i=d % self.dn
        self.__divide(s,f,i)
        mid=(s+f)//2
        self.area[mid]=np.array([s,f,i],dtype=int)
        self.__construct(d+1, s, mid)
        self.__construct(d+1, mid+1, f)
            
    def __divide(self,s,f,i):
        """
        divide the array by the median 
        into two classes
        
        Time complexity: O(n)
        
        Parameters
        ----------
 
        s : int
            start index
        
        f : int
            stop index(not included)
            
        i : which axis
            
        Returns
        -------
        None

        """
        Med=np.median(self.data[s:f,i])
        l=s
        r=f-1
        while l != r:
            if self.data[l,i] >= Med:
                temp=self.data[l,i].copy()
                self.data[l,i]=self.data[r,i]
                self.data[r,i]=temp
                r=r-1
            else:
                l=l+1
            
    def search(self,x,k):
        """
        

        Parameters
        ----------
        x : numpy.array
            target point
        k : int
            the number of the nearest
            points to find

        Returns
        -------
        the array of the nearest points

        """
        res=self.data[0:k].copy() #The result array
        dtmp=np.zeros((k,1))
        for t in range(0,k):
            dtmp[t]=LpDistance(x,res[t],self.p)
        res=np.concatenate((res,dtmp),axis=1)
        self.quicksort(res,0,k)
        self.__search(x,k,res,0,self.n_datas,0)
        return res
    
    def __search(self,x,k,res,s,f,d):
        #Notice the stop point is not included
        l,r=s,f
       # print("search start")
        Mid=[]
        mid=0
        dNow=d
        while l < r :
            #Find the leaf including x
            #Notice the left is not included 
            #the right is on the contrary
            i=dNow%self.dn
            dNow+=1
            mid=(l+r)//2
            if x[i] <= self.data[mid,i]:
                r=mid
                Mid.append((mid,0))
            else:
                l=mid+1
                Mid.append((mid,1))
        #The loop stop while delta=r-l=2 or 1
        #so the mid is the leaf
        leaf=mid
        ld=LpDistance(x,self.data[leaf],self.p)
        if ld < res[k-1,-1]:
            self.__insert(res,self.data[leaf],ld)
        dNow=len(Mid)+d-2 #depth at present
        while dNow >= d:
            radius=res[k-1,-1]
            m,flag=Mid[dNow-d] #m is the middle index now
            axis=dNow % self.dn
            tempP=np.zeros(self.dn)
            tempP[axis]=self.data[m,axis]-x[axis]
            ld=LpDistance(np.zeros(self.dn),tempP,self.p)
            if flag==0 and ld<radius:
                #Find if the right area possibly have closer point
                l=m+1
                r=self.area[m,1]
            if flag==1 and ld<radius:
                #in the left area
                l=self.area[m,0]
                r=m
            if ld<radius:
                self.__search(x,k,res,l,r,dNow)
            #if m is nearer
            ld=LpDistance(x,self.data[m],self.p)
            if ld<radius:
                self.__insert(res,self.data[m],ld)
          #  print("search end dnow:{0}".format(dNow))
            dNow-=1
            
            
    def __insert(self,arr,p,d):
        """
        insert a point into the array
        
        time complexity : O(k) 
        This can be optimized by using Linked list.

        Parameters
        ----------
        arr : arr
            points been sorted,the last column is distance
        p : numpy.array
            new point.
        d : float
            distance between p and x
        Returns
        -------
        None.

        """
       # print("insert start")
        k=len(arr)
        l=0
        r=k
        flag=0 #judge the last loop of search
        #Binary search
        while l<r:
            mid=(l+r)//2 #right median index
            if d == arr[mid,-1]:
                flag=0
                break
            if d < arr[mid,-1]:
                flag=-1
                r=mid
            else:
                flag=1
                l=mid+1
        #insert
        if flag<=0 :
            j=k-1
            while j>mid:
                arr[j]=arr[j-1]
                j-=1
            arr[mid]=np.append(p,d)
        else:
            j=k-1
            if mid+1==k:
                return
            while j>mid+1:
                arr[j]=arr[j-1]
                j-=1
            arr[mid+1]=np.append(p,d)
        #print("insert end")
            
    def quicksort(self,arr,l,r):
        """
        

        Parameters
        ----------
        x : np.arr
            target point
        arr : arr
            points array, the last column is distance
        l : int
            start index
        r : int
            stop index(not included)
             
        Returns
        -------
        None.

        """
       # print("quicksort start")
        if l>=r-1:
            return #no more than one point
        k=np.random.randint(l,r)
        temp=arr[r-1].copy()
        arr[r-1]=arr[k]
        arr[k]=temp
        dflag=arr[r-1,-1]
        i=l-1
        for p in range(l,r-1):
            if arr[p,-1]<=dflag:
                i+=1
                temp=arr[i].copy()
                arr[i]=arr[p]
                arr[p]=temp
        temp=arr[i+1].copy()
        arr[i+1]=arr[r-1]
        arr[r-1]=temp
        self.quicksort(arr,l,i+1)
        self.quicksort(arr,i+2,r)
       # print("quicksort end")

if(__name__ == "__main__"):
    dataArr,labelArr=loadData("../data/mnist_train.csv")
    dataTree=kdTree(dataArr)
    dataTree.construct()
    #test dataTree.construct
    for d in range(len(dataTree.area)):
        l,r,i=dataTree.area[d]
        mid=(l+r)//2
        if(l>=r-1):
            continue
        med=np.median(dataTree.data[l:r,i])
        for k in range(l,mid):
            if med <dataTree.data[k,i]:
                print("error1")
                break
        for k in range(mid+1,r):
            if med>dataTree.data[k,i]:
                print("error2")
                break
    x=np.random.rand(dataTree.dn)-10
    t1=time.time()
    res=dataTree.search(x,10)
    t2=time.time()
    print(t2-t1)
    N=0
    #test dataTree.search
    for y in dataTree.data:
        if LpDistance(x,y)<=res[8,-1]:
            N+=1
    if N!= 9:
        raise RuntimeError