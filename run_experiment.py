from train import main
import os
runs_per_condition = 3


parser = argparse.ArgumentParser(description='NN - Flo and Ant')
parser.add_argument('--folder', '-f', default="Experiments", help='path to folder')
parser.add_argument('--runs', '-r', default=3, help='how often the conditions is tested')
args = parser.parse_args()

folder_name = args.folder

if not os.path.exists(folder_name):
    os.mkdir(folder_name)


# Run i experiments
logs = []
for i in range(runs_per_condition):
    current_log = main(path = folder_name + "/", name = str(i))

    logs.append(current_log)
 
 
# Plot averaged accuracies and blue score:


# Create Attention Images for best model:
    
    

    
    

    
    
