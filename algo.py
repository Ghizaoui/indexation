import seaborn as sns
import numpy as np
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import glob
import cv2 as cv

def createCorellMatrix(x):
  val=np.unique(x);
  ng=len(val)
  n=len(x);
  #création et initiation du matrice
  corr=np.zeros((ng,ng));
  #création du matrice correlogramme
  for i in range(1,n-1):
    for j in range(1,n-1):
      a=x[i][j];
      for l in range(-1,2):
        for k in range(-1,2):
          if not(l==0 and k==0) :
            b=x[i+l][j+k];
            corr[a][b]=corr[a][b]+1;
  return corr;
    
#calcul du diagonale
def Diagonale(x):
  diag=x.diagonal();
  return diag;
  
  
#calcul du similarité
def Distance(H1,H2):
  minimum=0;
  somme=0;
  ng=len(H1);
  for i in range(ng):
    minimum=minimum+min(H1[i],H2[i]);
    somme=somme+H1[i];
  Dinter=1-(minimum/somme);
  return Dinter;


#ds = tfds.load("/content/dataset")
from tensorflow import keras







def calculDistance(H1,H2):
    distance =0     
    for i in range(len(H1)):
        distance += np.square(H1[i]-H2[i])
    return np.sqrt(distance)
    
def predict(path_test):
    #path = glob.glob("C:\\Users\\user\\Desktop\\hna\\dataset\\train\\fleur de lys\\*.*")
    path = glob.glob("dataset/train/fleur de lys/*.*")
    cv_img = []

    for img in path:
        n = cv.imread(img)
        n=cv.resize(n,(200,200))
        #convert 2d to 3d
        n=n.reshape(n.shape[0], (n.shape[1] * n.shape[2]))
        cv_img.append(n)
        
    a=len(cv_img)
    diag=[]
    for i in range(a):
      res=createCorellMatrix(cv_img[i]);
      diag.append(Diagonale(res));
    img_test= cv.imread(path_test);
    img_test=cv.resize(img_test,(200,200));
    #convert 2d to 3d
    img_test=img_test.reshape(img_test.shape[0], (img_test.shape[1] * img_test.shape[2]));
    corellMatrix_test=createCorellMatrix(img_test);
    diag_test=Diagonale(corellMatrix_test);
    ligne=[]
    col=[]
    for i in range(a):
      d=Distance(diag_test,diag[i]);
      #print("la distance entre l image "+str(path_test[0])+" et l image "+str(path[i])+" est "+str(d)+" ;")
      col.append((str(path[i]),str(d)));
    ligne.append(col);
    
    sorted_image_distance = sorted(ligne[0], key=lambda x: x[1])
    print(sorted_image_distance)

    return sorted_image_distance;
def calchisto(n):
        val=np.unique(n);
        #création et initiation du matrice
        #corr=np.zeros(256);
        corr=[]
        for i in range(256):
            corr.append(0);
        
        for j in val:
            cpt=0;
            for i in range(200):
                for l in range(200):
                    if n[i][l]==j:
                        cpt=cpt+1;
            
            corr[j]=cpt;    
        return corr



def predictH(path_test):
    path = glob.glob("dataset/train/fleur de lys/*.*")
    cv_img = []
    descriptors = []
    img_test = cv.imread(path_test)
    img_test=cv.resize(img_test,(200,200))
    #convert 2d to 3d
    img_test=img_test.reshape(img_test.shape[0], (img_test.shape[1] * img_test.shape[2]))
    hist_test=calchisto(img_test)

    for img in path:
        n = cv.imread(img)
        n=cv.resize(n,(200,200))
        #convert 2d to 3d
        n=n.reshape(n.shape[0], (n.shape[1] * n.shape[2]))
        corr=calchisto(n)             
        descriptors.append(corr)
    ligne=[]
    col=[]
    for i in range(len(descriptors)):
      d=Distance(hist_test,descriptors[i]);
      #print("la distance entre l image "+str(path_test[0])+" et l image "+str(path[i])+" est "+str(d)+" ;")
      col.append((str(path[i]),str(d)));
    ligne.append(col);
    sorted_image_distance = sorted(ligne[0], key=lambda x: x[1])
    return sorted_image_distance
def textcolor(path):
    t1=predict(path)
    t2=predictH(path)
    feature=np.concatenate((t1,t2))
    print(feature)
    return feature
    
    
