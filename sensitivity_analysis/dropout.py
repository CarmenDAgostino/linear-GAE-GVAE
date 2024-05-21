import subprocess
import re

# codice che effettua l'analisi di sensibilità sul parametro dropout

# variabili
output_file = "results/sa_dropout_results.txt"
train_command = "python ../linear_gae/train.py"

# parametri
dropouts = [0.0,0.1,0.3,0.5,0.7,0.9]
models = ["gcn_ae","gcn_vae","linear_ae","linear_vae"]
datasets = ["pubmed"]
tasks = ["link_prediction","node_clustering"]

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

        file.write(f"Dropout: {dim}\n")
        file.write(f"AUC scores: {auc_values}\n")
        file.write(f"AP scores: {ap_values}\n")
        file.write(f"Total running times: {time_values}\n\n")

    else : # task node_clustering
        mi_pattern = re.compile(r'Adjusted MI scores\s+\[(.*?)\]', re.DOTALL)
        time_pattern = re.compile(r'Total Running times\s+\[(.*?)\]', re.DOTALL)
        
        mi_match = mi_pattern.search(output)
        time_match = time_pattern.search(output)
        
        mi_values = mi_match.group(1) if mi_match else ""
        time_values = time_match.group(1) if time_match else ""

        file.write(f"Dropout: {dim}\n")
        file.write(f"AMI scores: {mi_values}\n")
        file.write(f"Total running times: {time_values}\n\n")

file = open(output_file, "a")

# training per diversi valori di dropout
for task in tasks:
    file.write(f"TASK: {task}\n\n" )
    for dataset in datasets:
        file.write(f"DATASET: {dataset}\n" )
        for model in models:
            file.write(f"MODEL: {model}\n" )
            for value in dropouts:

                print(f"Training and testing model {model} on dataset {dataset} and on task {task} with dropout {value}...")

                modified_command = f"{train_command} --model={model} --dataset={dataset} --task={task} --dropout={value} --kcore=True"
                output = subprocess.check_output(modified_command, shell=True) 
                output = output.decode()  

                print(f"Model trained and tested for dropout {value}.")
                
                write_output(output,task,value)

file.close()    

print("Sensitivity analysis on dropout completed. Results saved in", output_file)
