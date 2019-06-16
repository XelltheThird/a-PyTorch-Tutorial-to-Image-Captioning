from train import main
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from multiCaptionGenerator import genVis


parser = argparse.ArgumentParser(description='NN - Flo and Ant')
parser.add_argument('--folder', '-f', default="Experiments", help='path to folder')
parser.add_argument('--runs', '-r', default=3, help='how often the conditions is tested')
args = parser.parse_args()

folder_name = args.folder
runs = int(args.runs)

if not os.path.exists(folder_name):
    os.mkdir(folder_name)


# Run i experiments
logs = []
for i in range(runs):
    current_log = main(path = folder_name + "/", name = str(i))
    logs.append(current_log)

 
 
# Plot averaged accuracies and blue score:
logs_length = len(logs[0]['train_acc'])
best_bleu = 0
best_run = -1
for key in logs[0]:
    key_data = []
    for run in range(len(logs)):
        data = logs[run][key]
        key_data.append(data)
        
        # Evaluate quality of model
        if key == "val_bleu4":
            max_bleu = max(data)
            if max_bleu > best_bleu:
                best_bleu = max_bleu
                best_run = run
            
        
        
    data_mean = key_data.mean(axis=0)
    data_std = key_data.std(axis=0)
    lower_bound = data_mean - data_std
    upper_bound = data_mean + data_std
    
    # Plot stuff
    p = plt.plot(range(logs_length), data_mean, label=key)
    color = p[0].get_color()
    plt.fill_between(range(logs_length), lower_bound, upper_bound, alpha=0.5, color=color)
    
    
    

plt.legend()
plt.title('Evaluation metrics over training time')
plt.xlabel('Epochs')
plt.ylabel('Value of Metric')
plt.savefig(folder_name + "/" + "training_curves.pdf")
    

# Create Attention Images for best model:
best_model_name = "BEST_" + str(best_run) + ".pth.tar"
folder_name += '/' 
genVis(folder_name, best_model_name)
    

# Save logs to file:
import pickle
 
 
# Step 2
with open(folder_name + "pickled_logs.json", 'wb') as logs_file:
 
  # Step 3
  pickle.dump(logs, logs_file)
    

    
    
