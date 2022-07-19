import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from utils import load_fer
import cv2

from explain.DnCShap.utils.DncShap import DnCShap
from explain.DnCShap.utils.Plots import plot_shap



if __name__ == '__main__':    
    model = torch.load('./experiments/DACL/FER/exp_1/models/best_model.pth')

    test_data = load_fer.FERplus(0, idx_set=2,
                                                max_loaded_images_per_label=120,
                                                transforms=None,
                                                base_path_to_FER_plus='C:/Users/rielcheikh/Desktop/FER/DB/FER+/')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2, pin_memory = True)
    
    
    
    width= 96
    height= 96
    times = 2
    num_classes = 8
    
    classes = ['NEUTRAL','HAPPY','SAD','SURPRISE','FEAR','DISGUST','ANGER','CONTEMPT']
    
    
    for c in range(num_classes):
        count = 0
    for i, (images, target, _) in enumerate(test_loader):
        if i == 0:
            fig, ax = plt.subplots()#figsize=(9, 4))
            
            img = cv2.imread("C:/Users/rielcheikh/Desktop/FER/DB/FER+/Images/FER2013Test/fer0032220.png")
            print("-------",type(img), img.shape)
            cv2.imshow("s", img)
             
            cv2.waitKey(0) # waits until a key is pressed
            cv2.destroyAllWindows()
            
            img2 = ax.imshow(images.reshape(3,width,height),alpha= 0.4)
            
            plt.axis('off')
            
            print("Saved")
            fig.savefig('./explain/DnCShap/results/res_x/xx.png',format="png",dpi=600)
            plt.close(fig)