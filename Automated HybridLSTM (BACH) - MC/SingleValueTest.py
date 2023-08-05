from src import *
import numpy as np
from skopt import gp_minimize, forest_minimize, dummy_minimize, gbrt_minimize
import skopt

import torch
from torch.distributions import Categorical



args = ModelOptions().parse()

LABELS = ['Normal', 'Benign', 'InSitu', 'Invasive']
LabelAbbrev = ['n', 'b', 'is', 'iv']
        
        
def Two_Stage_Output (p1, threshold, mode = 'test'):
    
    #print('Threshold value: ', torch.Tensor(threshold))
    print('Threshold:', threshold)
    file.write("Threshold value = " + str(threshold))
    file.write("\n")
    NO_OF_MODELS = 1
    ens_results = []
    excluded_imgs = []   
    for i in range(len(p1)):
        chosen_models = []
        all_models = []
        
        #1
        uncert = np.mean(p1[i][1])
        all_models.append(np.array(p1[i][0]))
        if uncert < threshold:
            chosen_models.append(np.array(p1[i][0]))
        

        #----------------------------------------
        if len(chosen_models) != 0:
            final = np.sum(chosen_models, axis=0)
            final = np.divide(final, len(chosen_models))
            highest_index = np.argmax(final)
            label = LABELS[highest_index]
            image_name = p1[i][2]        
            ens_results.append([label, image_name, final, len(chosen_models)])
        
        if len(chosen_models) == 0:
            sum_all_models = np.sum(all_models, axis=0)
            final1 = np.divide(sum_all_models, NO_OF_MODELS)
            highest_index = np.argmax(final1)
            label = LABELS[highest_index]
            exc_image_name = p1[i][2]
            excluded_imgs.append([label, exc_image_name, final1, len(chosen_models)])  
         
 
    file.write('Model output:\n')
    file.write('--------------------------------------\n')
    for i in ens_results:
        file.write("%s\n" % i)
    
    val_acc = 0
    if mode == 'valid':
        correct = 0
        for i in range(len(ens_results)):
            if ens_results[i][0] == LABELS[0] and LabelAbbrev[0] in ens_results[i][1]:
                correct+=1
            elif ens_results[i][0] == LABELS[1] and LabelAbbrev[1] in ens_results[i][1]:
                correct+=1
            elif ens_results[i][0] == LABELS[2] and LabelAbbrev[2] in ens_results[i][1]:
                correct+=1
            elif ens_results[i][0] == LABELS[3] and LabelAbbrev[3] in ens_results[i][1]:
                correct+=1
            
            

        file.write('Included Images: %d\n' % (len(ens_results)))        
        file.write('Excluded Images: %d\n' % (len(excluded_imgs)))

        
        if len(ens_results) != 0:
            val_acc = (correct / len(ens_results)) *100
        file.write('Validation Accuracy = %f %%\n' % val_acc)


    file.write('EXCLUDED IMGS\n')
    file.write('---------------------\n')
    for j in excluded_imgs:
        file.write("%s\n" % j)
    
    exc_val_acc = 0
    if mode == 'valid':
        correct = 0
        for i in range(len(excluded_imgs)):
            if excluded_imgs[i][0] == LABELS[0] and LabelAbbrev[0] in excluded_imgs[i][1]:
                correct+=1
            elif excluded_imgs[i][0] == LABELS[1] and LabelAbbrev[1] in excluded_imgs[i][1]:
                correct+=1
            elif excluded_imgs[i][0] == LABELS[2] and LabelAbbrev[2] in excluded_imgs[i][1]:
                correct+=1
            elif excluded_imgs[i][0] == LABELS[3] and LabelAbbrev[3] in excluded_imgs[i][1]:
                correct+=1
            
                
        if len(excluded_imgs) != 0:
            exc_val_acc = (correct / len(excluded_imgs)) *100
        file.write('-----------------------\n')
        file.write('Exc Validation Accuracy = %f %%\n' % exc_val_acc)
    file.write('-----------------------\n')
    

    
    
    

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


pw_network = networks.PatchWiseNetwork(args.channels)
iw_network = networks.ImageWiseNetwork(args.channels)


if args.testset_path is '':
    import tkinter.filedialog as fdialog

    args.testset_path = fdialog.askopenfilename(initialdir=r"./dataset/test", title="choose your file", filetypes=(("tiff files", "*.tif"), ("all files", "*.*")))

if args.network == '2':
    

    im_model = models.ImageWiseModel(args, iw_network, pw_network)
    pred = im_model.test(args.testset_path, ensemble= False)
    
    Test_Fold = args.Tfold
    #file = open(r'G:/Automated Two Stage CNN (Brain)/ReportsT_F1/Forest/BestBetas_ExcImgs_TF4_VF5_Forest.txt', 'w')
    file = open(r'G:/Automated HybridLSTM (BACH) - MC/ReportsT_'+Test_Fold+'/Test_'+Test_Fold+'_all_methods_extended.txt', 'w')

    BetasF1 = [0.03031869, 0.01, 0.0359344, 0.92011183, 2.5]
    BetasF2 = [0.03508988, 0.01, 0.03555734, 0.07085321, 2.5]
    BetasF3 = [0.03070367, 0.01, 0.03422032, 0.29, 2.5]
    BetasF4 = [0.031239329, 0.01, 0.037342, 0.3224999, 2.5]
    BetasF5 = [0.0330441755, 0.01, 0.3625963, 0.08816, 2.5]
    
    if Test_Fold == 'F1':
        Betas = BetasF1
    elif Test_Fold == 'F2':
        Betas = BetasF2
    elif Test_Fold == 'F3':
        Betas = BetasF3
    elif Test_Fold == 'F4':
        Betas = BetasF4
    elif Test_Fold == 'F5':
        Betas = BetasF5
        
    ops = ['GP', 'COBYLA', 'DualAnn', 'NSGA2', 'No Opt']
    n = 0
    for x in Betas:
        file.write('Optimizer:'+ str(ops[n]))
        file.write('\n')
        y = Two_Stage_Output(pred, x, mode = 'valid')
        #print('Avg class rate: ', y)
        file.write('Avg class rate: '+ str(y))
        file.write('\n')
        n = n+1

    
else:
    
    pw_model = models.PatchWiseModel(args, pw_network)
    pred = pw_model.test(args.testset_path)



