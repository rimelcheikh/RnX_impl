import torch.nn.functional as F
from methods.GradCAM.utils.gradcam import output_last_conv, get_grad, avg_grad, conv_x_avg_grads, avg_output, display_in_original_space

def GradCAM_runner(test_loader, model, classes, save_path):
    corrects = True
    
    for c in range(8):
        count = 0
        for i, (images, target, _) in enumerate(test_loader):
                img = images
        
                pred, conv_model, img = output_last_conv(model, img)
                
                if(corrects):
                    if(F.softmax(pred, dim=-1).argmax().item() == target.item() == c):
                        
                        grads = get_grad(pred, conv_model)
                        pooled_grads = avg_grad(grads)
                        conv_output = conv_x_avg_grads(conv_model, pooled_grads)
                        heatmap = avg_output(conv_output)
                        #display_in_downsized_space(heatmap.cpu().detach())
                        display_in_original_space(heatmap.cpu().detach(),img,save_path+"/"+classes[target.item()]+"_"+str(count)+".png")
                    
                        count+=1
                    """if(count==10):
                      break"""
                  
                else:
                    if(F.softmax(pred, dim=-1).argmax().item() != target.item() == c):
                    
                        grads = get_grad(pred, conv_model)
                        pooled_grads = avg_grad(grads)
                        conv_output = conv_x_avg_grads(conv_model, pooled_grads)
                        heatmap = avg_output(conv_output)
                        #display_in_downsized_space(heatmap.cpu().detach())
                        display_in_original_space(heatmap.cpu().detach(),img,save_path+"/"+classes[target.item()]+"_"+classes[F.softmax(pred, dim=-1).argmax().item()]+"_"+str(count)+".png")
                    
                        count+=1
                    """if(count==10):
                      break"""