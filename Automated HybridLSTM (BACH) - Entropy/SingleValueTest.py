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
        p_tensor = torch.Tensor(np.array(p1[i][0]))
        entropy = Categorical(probs = p_tensor).entropy() 
        all_models.append(np.array(p1[i][0]))
        if entropy.numpy() < threshold:
            chosen_models.append(np.array(p1[i][0]))
        

        #----------------------------------------
        if len(chosen_models) != 0:
            final = np.sum(chosen_models, axis=0)
            final = np.divide(final, len(chosen_models))
            highest_index = np.argmax(final)
            label = LABELS[highest_index]
            image_name = p1[i][1]        
            ens_results.append([label, image_name, final, len(chosen_models)])
        
        if len(chosen_models) == 0:
            sum_all_models = np.sum(all_models, axis=0)
            final1 = np.divide(sum_all_models, NO_OF_MODELS)
            highest_index = np.argmax(final1)
            label = LABELS[highest_index]
            exc_image_name = p1[i][1]
            excluded_imgs.append([label, exc_image_name, final1, len(chosen_models)])  
         
 
    file.write('Hybrid LSTM output:\n')
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
    
    file.write('-----------------------\n')
    
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


    if len(excluded_imgs) != 0:
        classif_rate_exc = 0
        for i in range(len(excluded_imgs)):
            if LabelAbbrev[0] in excluded_imgs[i][1]:
                Q = np.array([1, 0, 0, 0])
            elif LabelAbbrev[1] in excluded_imgs[i][1]:
                Q = np.array([0, 1, 0, 0])
            elif LabelAbbrev[2] in excluded_imgs[i][1]:
                Q = np.array([0, 0, 1, 0])
            elif LabelAbbrev[3] in excluded_imgs[i][1]:
                Q = np.array([0, 0, 0, 1])

                
            out_array_ex = np.multiply(excluded_imgs[i][2], Q)

            classif_rate_exc = classif_rate_exc + sum(sum(out_array_ex))
            
        average_class_rate_exc = np.divide(classif_rate_exc, len(excluded_imgs))

    elif len(excluded_imgs) == 0:
        average_class_rate_exc = 1
 
        
    if len(ens_results) != 0:
        classif_rate_inc = 0
        for i in range(len(ens_results)):
            if LabelAbbrev[0] in ens_results[i][1]:
                T = np.array([1, 0, 0, 0])
            elif LabelAbbrev[1] in ens_results[i][1]:
                T = np.array([0, 1, 0, 0])
            elif LabelAbbrev[2] in ens_results[i][1]:
                T = np.array([0, 0, 1, 0])
            elif LabelAbbrev[3] in ens_results[i][1]:
                T = np.array([0, 0, 0, 1])
 
                
            out_array_inc = np.multiply(np.array(ens_results[i][2]), T)       
            classif_rate_inc = classif_rate_inc + sum(sum(out_array_inc))
       
        average_class_rate_inc = 1 - np.divide(classif_rate_inc, len(ens_results))
            
    elif len(ens_results) == 0:
        average_class_rate_inc = 1      
        
    AvgClassRate = average_class_rate_exc + average_class_rate_inc    
    return AvgClassRate
    
    

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

    file = open(r'G:/Automated HybridLSTM (BACH) - entropy/ReportsT_F5/Test_F5_all_methods_extended.txt', 'w')

    BetasF1 = [1.27102, 1.150625, 1.2867, 1.2433, 2.5]
    BetasF2 = [1.27752, 1.135, 1.27005, 1.2544, 2.5]
    BetasF3 = [1.27772, 1.35375, 1.32798, 1.18512, 2.5]
    BetasF4 = [1.217278, 1.2053125, 1.30052, 1.23487, 2.5]
    BetasF5 = [1.3086, 1.1975, 1.28646, 1.29174, 2.5]
    
    ops = ['GP', 'COBYLA', 'DualAnn', 'NSGA2', 'No Opt']
    n = 0
    for x in BetasF5:
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

