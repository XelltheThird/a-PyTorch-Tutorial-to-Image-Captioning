from train import main
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse



parser = argparse.ArgumentParser(description='NN - Flo and Ant')
parser.add_argument('--folder', '-f', default="Experiments", help='path to folder')
parser.add_argument('--runs', '-r', default=3, help='how often the conditions is tested')
args = parser.parse_args()

folder_name = args.folder
runs = args.runs

if not os.path.exists(folder_name):
    os.mkdir(folder_name)


# Run i experiments
logs = []
for i in range(runs):
    current_log = main(path = folder_name + "/", name = str(i))
    logs.append(current_log)
 
 
# Plot averaged accuracies and blue score:

logs_length = len(logs[0]['train_acc'])

for key in logs[0]:
    key_data = np.zeros((runs, logs_length))
    for run in logs:
        data = logs[key]
        key_data[run] = data
    data_mean = key_data.mean(axis=0)
    data_std = key_data.std(axis=0)
    lower_bound = data_mean - data_std
    upper_bound = data_mean + data_std
    
    
    p = plt.plot(range(logs_length), data_mean, label=key)
    color = p[0].get_color()
    plt.fill_between(range(logs_length), lower_bound, upper_bound, alpha=0.5, color=color)

plt.legend()
plt.title('Evaluation metrics over training time')
plt.xlabel('Epochs')
plt.ylabel('Value of Metric')
plt.savefig("Metrics.pdf")
    

# Create Attention Images for best model:
    
    

    
    

    
    
