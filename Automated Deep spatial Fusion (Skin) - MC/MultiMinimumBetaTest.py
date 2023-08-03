from src import *

import numpy as np
from skopt import gp_minimize, forest_minimize, dummy_minimize, gbrt_minimize
import skopt
import scipy.optimize as scop
from scipy.optimize import Bounds
import torch
from torch.distributions import Categorical

from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2

args = ModelOptions().parse()

LABELS = ['Benign', 'Malignant']
LabelAbbrev = ['B', 'M']
        
        
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
         
 
    file.write('Two Stage CNN output:\n')
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
        if len(excluded_imgs) != 0:
            exc_val_acc = (correct / len(excluded_imgs)) *100
        file.write('-----------------------\n')
        file.write('Exc Validation Accuracy = %f %%\n' % exc_val_acc)
    
    file.write('-----------------------\n')

    


    if len(excluded_imgs) != 0:
        classif_rate_exc = 0
        for i in range(len(excluded_imgs)):
            if LabelAbbrev[0] in excluded_imgs[i][1]:
                Q = np.array([1, 0])
            elif LabelAbbrev[1] in excluded_imgs[i][1]:
                Q = np.array([0, 1])

                
            out_array_ex = np.multiply(excluded_imgs[i][2], Q)

            classif_rate_exc = classif_rate_exc + sum(sum(out_array_ex))
            
        average_class_rate_exc = np.divide(classif_rate_exc, len(excluded_imgs))

    elif len(excluded_imgs) == 0:
        average_class_rate_exc = 1
 
        
    if len(ens_results) != 0:
        classif_rate_inc = 0
        for i in range(len(ens_results)):
            if LabelAbbrev[0] in ens_results[i][1]:
                T = np.array([1, 0])
            elif LabelAbbrev[1] in ens_results[i][1]:
                T = np.array([0, 1])
 
                
            out_array_inc = np.multiply(np.array(ens_results[i][2]), T)       
            classif_rate_inc = classif_rate_inc + sum(sum(out_array_inc))
       
        average_class_rate_inc = 1 - np.divide(classif_rate_inc, len(ens_results))
            
    elif len(ens_results) == 0:
        average_class_rate_inc = 1      
        
    return average_class_rate_exc, average_class_rate_inc    
    
    
    
class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=1,
                         n_obj=2,
                         xl=np.array([0]),
                         xu=np.array([2]),
                         elementwise_evaluation = True)

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = Two_Stage_Output(pred, X, mode = 'valid')      
        out["F"] = np.array(f1) 
        
        
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
    
    Test_Fold = 'F5'
    valid_Fold = 'F4'
    trials = [1,2,3,4,5]
    
    for i in trials:
        file = open(r'G:/Automated Deep spatial Fusion (skin)/ReportsT_'+Test_Fold+'/NSGA2/ExcImgs_T'+Test_Fold+'_V'+valid_Fold+'_NSGA2_trial'+str(i)+'.txt', 'w')
                    
        myproblem = MyProblem()
        algorithm = NSGA2(pop_size = 1)
        stop_criteria = ('n_gen', 50)
        
        res = minimize(problem = myproblem,
               algorithm = algorithm,
               termination = stop_criteria)
    
                
        file.write('Best Case:\n')
    
        file.write("AvgClassRate = " + str(res.F))
    
        file.write("\n")
        file.write("Threshold = " + str(res.X))
        file.close()
        print('end of trial', i)                
           
        
    
else:
    
    pw_model = models.PatchWiseModel(args, pw_network)
    pred = pw_model.test(args.testset_path)
