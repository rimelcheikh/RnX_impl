import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
from torch import autograd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow import keras


device = torch.device('cuda')
cudnn.benchmark = True

#1. Capture Output Of Last Convolution Layer

class LastConvLayerModel(nn.Module):
    def __init__(self, model):
        super(LastConvLayerModel, self).__init__()
        self.layers = list(list(model.children())[0].children())

    def forward(self, X_batch):
        
        x = self.layers[0](X_batch)
        
        conv_layer_output = None
        d = self.layers[1:8]+self.layers[10:12]
                       
        for i, layer in enumerate(d):
            if i == 8:
                x = torch.flatten(x, 1)
            x = layer(x)

            if i == 6: ## Output after last Convolution layer
                for i_last, layer_last in enumerate(list(layer.children())[1].children()):
                    if i_last == 3:
                        self.conv_layer_output = x
        return x
    
def output_last_conv(model, img):
    conv_model = LastConvLayerModel(model)
    model.eval()

    images = img.to(device)
    pred = conv_model(images)
    #print(i, F.softmax(pred, dim=-1).argmax(), F.softmax(pred, dim=-1).max())
    
    return pred, conv_model, images


#2. Take Gradients Of Last Conv Layer Output With Respect to Prediction
def get_grad(pred, conv_model):
    #The first value is the predicted probability (obtained with the last layer of the og model) 
    #and the second value is the output of the last convolution layer
    grads = autograd.grad(pred[:, pred.argmax().item()], conv_model.conv_layer_output)
    return grads
    
    

#3. Average Gradients
def avg_grad(grads):
    #averaged (summarized) gradients at the output channel level to get average gradients per channel
    #each value tells us how much each of the 512 feature maps(channels) influence the gradient
    pooled_grads = grads[0].mean((0,1))  #summarized values of the gradients in the backward pass
    return pooled_grads


#4. Multiply Convolution Layer Output with Averaged Gradients
def conv_x_avg_grads(conv_model,pooled_grads):
    #multiply the output of the last convolution layer with averaged gradients
    #to see how imporant each of the og feature planes in regard to the predicted class
    conv_output = conv_model.conv_layer_output.squeeze()  #og activatiosn of the forward pass
    #conv_output = F.relu(conv_output)
    for i in range(len(pooled_grads)):
        conv_output[i,:,:] *= pooled_grads[i]
    conv_output = F.relu(conv_output)
    return conv_output
    
    
#5. Average Output At Channel Axis To Create Heatmap
def avg_output(conv_output):
    # compute the average at the channel level on the output of the previous step
    #This will generate a heatmap of shape which will have activations that contribute to the predictions
    heatmap = conv_output.mean(dim=0).squeeze()  #mean along channels 
    heatmap = heatmap / torch.max(heatmap)  #normalizes the heatmap
    return heatmap


def display_in_downsized_space(heatmap):
    cmap = matplotlib.cm.get_cmap("Reds")

    fig = plt.figure(figsize=(3,3))

    ax2 = fig.add_subplot(122)
    ax2.imshow(heatmap, cmap="Reds")
    ax2.set_title("Gradients")
    ax2.set_xticks([],[])
    ax2.set_yticks([],[])


def display_in_original_space(heatmap, img_path, save_path, alpha=1):
    
    img = img_path.squeeze(0)
    img = img.cpu().detach() # keras.preprocessing.image.load_img(img_path)
    img = T.ToPILImage()(img)
    img = keras.preprocessing.image.img_to_array(img)
    
    print(img.shape)

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

    # Save the superimposed image
    superimposed_img.save(save_path)

    # Display Grad CAM
    #display(Image("./explain/grad-cam/res/res.png"))