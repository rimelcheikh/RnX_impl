import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 

import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

import matplotlib.cm as cm
import torch
import torchvision.transforms as T
import cv2



def plot_shap(model_main,img_data,labels,classes,data,idx,width=256,height=256,true_only=False,percentile=0,ax=None,fig=None,to_save=False,fname=""):

  if(ax==None and fig ==None):
    fig, ax= plt.subplots()#figsize=(8, 4))

  data1 = data.copy()
  data2 = abs(data1)
  value = np.percentile(data2,percentile)
  data1[abs(data1) < value] = 0
  norm = np.linalg.norm(data1)
  data1= data1/norm

  abs_max = np.percentile(np.abs(data1), 100)
  abs_min = abs_max

  cmap = 'seismic'
  label = labels[idx]
  
  _, output, _ = model_main(img_data)
  pred = torch.argmax(output, dim=1)
  
  pred = classes[pred]
  true = classes[label]
  print("True label : " + str(true)+" Predicted label : "+str(pred))
  

  if(true_only==True):
    data1[data1<0] = 0

   
  img = img_data.squeeze(0)
  img = img.cpu().detach() # keras.preprocessing.image.load_img(img_path)
  img = T.ToPILImage()(img)
  img = keras.preprocessing.image.array_to_img(img)
  
  img1 = ax.imshow(data1, interpolation='none', cmap = cmap, vmin=-abs_min, vmax=abs_max)
  img2 = ax.imshow(img,alpha=0.6)
  
  fig.colorbar(img1, ax=ax)
  plt.axis('off')
  if(to_save):
    print("Saved")
    fig.savefig(fname,format="png")#,dpi=600)
    plt.close(fig)

def display_gradcam(img, heatmap, alpha=0.003,to_save=False,fname=""):
    fig, ax= plt.subplots(figsize=(8, 4))

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Display Grad CAM
    x = ax.imshow(superimposed_img)

    fig.colorbar(cm.ScalarMappable( cmap="jet"), ax=ax)
    plt.axis('off')
    if(to_save):
      print("Saved")
      fig.savefig(fname,format="pdf",dpi=600)
      plt.close(fig)


def confusion_matrix_save(model_main,X_test,y_test,num_classes,classes,if_save=False,file_name=""):

    y_prob = model_main.predict(X_test)
    y_pred = [np.argmax(prob) for prob in y_prob]
    y_true = [np.argmax(prob) for prob in y_test]
    confusion_matrix1 = confusion_matrix(y_true,y_pred).astype("float32")
    print(confusion_matrix1.sum(axis=1))
    x = confusion_matrix1.sum(axis=1).reshape(7,1)
    confusion_matrix1= (confusion_matrix1/x)*100
    print(confusion_matrix1.sum(axis=1))
    nb_classes = num_classes

    fig, ax = plt.subplots(figsize=(15,10))
    classes = classes
    df_cm = pd.DataFrame(confusion_matrix1, index=classes, columns=classes)
    heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f",ax=ax)

    sns.set(font_scale=2)

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)
    if(if_save):
        fig.savefig(file_name,format="pdf",dpi=600,pad_inches = 80)