import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor


def RecShap(model,
            data,
            width,
            height,
            x_s, 
            x_e, #x_s and x_e initial and final dimensions for width
            y_s,
            y_e, #y_s and y_e initial and final dimensions for height
            pred_b, #prediction with both parts perturbed
            pred_f, #prediction with no parts perturbed
            score, #SHAP value of the part
            arg_max, #predicted label
            times, #number of parts the image is to be divided
            i,
            value #activation map
            ):

  if(times==0):
    value[x_s:x_e,y_s:y_e] = score.item()
    return

  else:

    if(i==1):
      
      pred_b = pred_b
      arg_max = arg_max
      times = times

      data_1 = np.zeros([3,width,height])
      data_2 = np.zeros([3,width,height])

      y_m = (y_s + y_e)//2

      data_1[:,:,:y_m] = data[:,:,:y_m]
      data_2[:,:,y_m:] = data[:,:,y_m:]

      data_1 = data_1.reshape(1,3,width,height)
      data_2 = data_2.reshape(1,3,width,height)

      _, pred_1, _ = model(torch.from_numpy(data_1).type(torch.cuda.FloatTensor))
      pred_1 = pred_1[0][arg_max]
      
      _, pred_2, _ = model(torch.from_numpy(data_2).type(torch.cuda.FloatTensor))
      pred_2 = pred_2[0][arg_max]
      
      score_1 = (((pred_1-pred_b) + (pred_f-pred_2))/2)/2
      score_2 = (((pred_2-pred_b) + (pred_f-pred_2))/2)/2

      times-=1
      RecShap(model,data,width,height,x_s,x_e,y_s,y_m,pred_b+pred_2,pred_1+pred_f,score_1,arg_max,times,0,value)
      RecShap(model,data,width,height,x_s,x_e,y_m,y_e,pred_b+pred_1,pred_2+pred_f,score_2,arg_max,times,0,value)

    if(i==0):
    
      pred_b = pred_b
      arg_max = arg_max
      times = times

      data_1 = np.zeros([3,width,height])
      data_2 = np.zeros([3,width,height])

      x_m = (x_s + x_e)//2

      data_1[:,:x_m,:] = data[:,:x_m,:]
      data_2[:,x_m:,:] = data[:,x_m:,:]

      data_1 = data_1.reshape(1,3,width,height)
      data_2 = data_2.reshape(1,3,width,height)


      _, pred_1, _ = model(torch.from_numpy(data_1).type(torch.cuda.FloatTensor))
      pred_1 = pred_1[0][arg_max]
      
      _, pred_2, _ = model(torch.from_numpy(data_2).type(torch.cuda.FloatTensor))
      pred_2 = pred_2[0][arg_max]

      score_1 = (((pred_1-pred_b) + (pred_f-pred_2))/2)/2
      score_2 = (((pred_2-pred_b) + (pred_f-pred_2))/2)/2

      times-=1

      #print(type(model),type(data),type(width),type(height),type(x_s),type(x_m),type(y_s),type(y_e),type(pred_b+pred_2),type(pred_1+pred_f),type(score_1),type(arg_max),type(times),type(1),type(value))
      RecShap(model,data,width,height,x_s,x_m,y_s,y_e,pred_b+pred_2,pred_1+pred_f,score_1,arg_max,times,1,value)
      RecShap(model,data,width,height,x_m,x_e,y_s,y_e,pred_b+pred_1,pred_2+pred_f,score_2,arg_max,times,1,value)



def DnCShap(model,data,width,height,times):
   
  data_b = np.zeros([3,width,height])
  data_f = data
  data_1 = np.zeros([3,width,height])
  data_2 = np.zeros([3,width,height])
  
  x_m = width//2
  y_m = height//2
   
  data_1[:,0:x_m,:] = data[:,0:x_m,:]
  data_2[:,x_m:,:] = data[:,x_m:,:]

  data_f = data_f.reshape(1,3,width,height)
  data_1 = data_1.reshape(1,3,width,height)
  data_2 = data_2.reshape(1,3,width,height)
  data_b = data_b.reshape(1,3,width,height)

  _, pred, _ = model(torch.from_numpy(data_f))
  arg_max = torch.argmax(pred, dim=1)
  pred_f = pred[0][arg_max]
  
  _, pred_b, _ = model(torch.from_numpy(data_b).type(torch.cuda.FloatTensor))
  pred_b = pred_b[0][arg_max]
  
  _, pred_1, _ = model(torch.from_numpy(data_1).type(torch.cuda.FloatTensor))
  pred_1 = pred_1[0][arg_max]
  
  _, pred_2, _ = model(torch.from_numpy(data_2).type(torch.cuda.FloatTensor))
  pred_2 = pred_2[0][arg_max]

  score_1 = ((pred_1-pred_b) + (pred_f-pred_2))/2
  score_2 = ((pred_2-pred_b) + (pred_f-pred_1))/2

  shap_value = np.zeros([width,height])


  times-=1
  RecShap(model,
          data,
          width,
          height,
          0,
          x_m,
          0,
          height,
          (pred_b+pred_2),
          (pred_1+pred_f),
          score_1,
          arg_max,
          times,
          1,
          shap_value)
  RecShap(model,
          data,
          width,
          height,
          x_m,
          width,
          0,
          height,
          (pred_b+pred_1),
          (pred_2+pred_f),
          score_2,
          arg_max,
          times,
          1,
          shap_value)


  return shap_value



