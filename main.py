from argparse import ArgumentParser
from methods.DnCShap.run_dncshap import DnCShap_runner
from methods.GradCAM.run_gradcam import GradCAM_runner
import torch
from torch.utils.data import DataLoader
from load_dataset import load_fer, load_an, load_ck
import sys
import os


parser = ArgumentParser()

parser.add_argument('--ds', type=str, default='fer')
parser.add_argument('--model', type=str, default='dacl')
parser.add_argument('--method', type=str, default='dncshap')
parser.add_argument('--save_path', type=str, default='exp_x')



if __name__ == '__main__': 
    
    args = parser.parse_args()
    print("Running "+ args.method + " for "+args.model+" trained on "+args.ds)


    classes = ['NEUTRAL','HAPPY','SAD','SURPRISE','FEAR','DISGUST','ANGER','CONTEMPT']
    
    try:
        model = torch.load('./models/'+args.model+'_'+args.ds+'.pth')
    except:
        print("Error in dataset or model name")
        sys.exit(1)
    
    
    try:
        if (args.ds == 'fer'):
            test_data = load_fer.FERplus(0, 
                                         idx_set=2,
                                         max_loaded_images_per_label=1000000,
                                         transforms=None,
                                         base_path_to_FER_plus='C:/Users/rielcheikh/Desktop/FER/DB/FER+/')
        elif (args.ds == 'an'):
            test_data = "to_do"
            
        elif (args.ds == 'ck'):
            test_data = "to_do"
        
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2, pin_memory = True)

            
    except:
        print("Dataset is not handeled")
        sys.exit(1)
    
    
    
    try:
        if not os.path.isdir('./results/'+args.method+'/'+args.model+'/'+args.ds+'/'+args.save_path):
            os.makedirs('./results/'+args.method+'/'+args.model+'/'+args.ds+'/'+args.save_path)

        
        if (args.method == 'dncshap'):
            DnCShap_runner(test_loader, model, classes, './results/'+args.method+'/'+args.model+'/'+args.ds+'/'+args.save_path)
            
        elif (args.method == 'gradcam'):
            GradCAM_runner(test_loader, model, classes, './results/'+args.method+'/'+args.model+'/'+args.ds+'/'+args.save_path)

            
    except:
        print("Method is not handeled")
        sys.exit(1)