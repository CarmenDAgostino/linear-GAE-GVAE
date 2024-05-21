import subprocess
import re

# codice che effettua l'analisi di sensibilità sui parametri che determinano 
# il numero di link rimossi nel task di link prediction

# variabili
output_file = "results/sa_link_num_results.txt"
train_command = "python ../linear_gae/train.py"

# parametri
link_num = [5.,10.,15.,20.,25.,30.,35.,40.,45.,50.]
models = ["gcn_ae","gcn_vae","linear_ae","linear_vae"]
datasets = ["pubmed"]
tasks = ["link_prediction"]

# funzione per scrivere il risultato
def write_output(output,task,dim) :

    if task == "link_prediction" :
        # Espressioni regolari per estrarre i risultati
        auc_pattern = re.compile(r'AUC scores\s+\[(.*?)\]', re.DOTALL)
        ap_pattern = re.compile(r'AP scores\s+\[(.*?)\]', re.DOTALL)
        time_pattern = re.compile(r'Total Running times\s+\[(.*?)\]', re.DOTALL)
        
        auc_match = auc_pattern.search(output)
        ap_match = ap_pattern.search(output)
        time_match = time_pattern.search(output)
        
        auc_values = auc_match.group(1) if auc_match else ""
        ap_values = ap_match.group(1) if ap_match else ""
        time_values = time_match.group(1) if time_match else ""

        file.write(f"Percentage of links removed: {dim}\n")
        file.write(f"AUC scores: {auc_values}\n")
        file.write(f"AP scores: {ap_values}\n")
        file.write(f"Total running times: {time_values}\n\n")

    else :
        mi_pattern = re.compile(r'Adjusted MI scores\s+\[(.*?)\]', re.DOTALL)
        time_pattern = re.compile(r'Total Running times\s+\[(.*?)\]', re.DOTALL)
        
        mi_match = mi_pattern.search(output)
        time_match = time_pattern.search(output)
        
        mi_values = mi_match.group(1) if mi_match else ""
        time_values = time_match.group(1) if time_match else ""

        file.write(f"Percentage of links removed: {dim}\n")
        file.write(f"AMI scores: {mi_values}\n")
        file.write(f"Total running times: {time_values}\n\n")

file = open(output_file, "a")

# training per diversi valori di dimansion
for task in tasks:
    file.write(f"TASK: {task}\n\n" )
    for dataset in datasets:
        file.write(f"DATASET: {dataset}\n" )
        for model in models:
            file.write(f"MODEL: {model}\n" )
            for dim in link_num:

                test_percent = (2/3) * dim
                val_percent = (1/3) * dim
                
                print(f"Training and testing model {model} on dataset {dataset} and on task {task} with percentage of links removed {dim}...")

                modified_command = f"{train_command} --model={model} --dataset={dataset} --task={task} --prop_test={test_percent} --prop_val={val_percent} --features=True --kcore=True"
                output = subprocess.check_output(modified_command, shell=True) 
                output = output.decode()  

                print(f"Model trained and tested with a percentage of removed link of {dim}.")
                
                write_output(output,task,dim)

file.close()    

print("Sensitivity analysis on percentage of removed link completed. Results saved in", output_file)
