import torch 
from methods.DnCShap.utils.DncShap import DnCShap
from methods.DnCShap.utils.Plots import plot_shap


def DnCShap_runner(test_loader, model, classes, save_path, times=4):
    print("ok")
    num_classes = 8
   
    corrects = True
    
    for c in range(num_classes):
        count = 0
        for i, (images, target, _) in enumerate(test_loader):
                height = images.size()[2]
                width = images.size()[3]
                
                _, output, _ = model(images)
                pred = torch.argmax(output, dim=1)
                
                if(corrects):
                    if(pred.item() == target.item() == c):
                        pred = classes[pred.item()]
                        true = classes[target.item()]
                        shap_value = DnCShap(model,images.squeeze(0).numpy(),width,height,times)
                        
                        plot_shap(model,
                                  images.to(torch.device('cuda')),
                                  target,
                                  classes,
                                  shap_value, 
                                  0, 
                                  width=96, height=96,
                                  true_only =True, 
                                  percentile=70,
                                  to_save=True,
                                  fname =save_path+"/"+true+"_"+str(count)+".png")
                        count+=1
                    if(count==10):
                      break
                  
                else:
                     if(pred.item() != target.item() == c):
                        pred = classes[pred.item()]
                        true = classes[target.item()]
                        shap_value = DnCShap(model,images.squeeze(0).numpy(),width,height,times)
                        
                        plot_shap(model,
                                  images.to(torch.device('cuda')),
                                  target,
                                  classes,
                                  shap_value, 
                                  0, 
                                  width=96, height=96,
                                  true_only =True, 
                                  percentile=70,
                                  to_save=True,
                                  fname =save_path+"/"+true+"_"+pred+"_"+str(count)+".png")
                        count+=1
                     