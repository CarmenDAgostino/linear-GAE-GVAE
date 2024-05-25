import subprocess
import re

# codice che effettua l'analisi di sensibilità sui parametri del task di node clustering 
# in particolare esegue il clustering dei nodi usando diversi algoritmi di clustering 

# variabili
output_file = "results/sa_node_clustering_results.txt"
train_command = "python ../linear_gae/train_modified.py"

# parametri
clustering_alg = ["kmeans","dbscan","agglomerative"]
models = ["gcn_ae","gcn_vae","linear_ae","linear_vae"]
datasets = ["cora","citeseer","pubmed"]
tasks = ["node_clustering"]


# funzione per scrivere il risultato
def write_output(output,alg) :

    mi_pattern = re.compile(r'Adjusted MI scores\s+\[(.*?)\]', re.DOTALL)
    s_pattern = re.compile(r'Silhouette scores\s+\[(.*?)\]', re.DOTALL)
    time_pattern = re.compile(r'Total Running times\s+\[(.*?)\]', re.DOTALL)
    
    mi_match = mi_pattern.search(output)
    s_match = s_pattern.search(output)
    time_match = time_pattern.search(output)
    
    mi_values = mi_match.group(1) if mi_match else ""
    s_values = s_match.group(1) if s_match else ""
    time_values = time_match.group(1) if time_match else ""

    file.write(f"Clustering algorithm: {alg}\n")
    file.write(f"AMI scores: {mi_values}\n")
    file.write(f"Silhouette scores: {s_values}\n")
    file.write(f"Total running times: {time_values}\n\n")


file = open(output_file, "a")

# training per diversi algoritmi di clustering
for task in tasks:
    file.write(f"TASK: {task}\n\n" )
    for dataset in datasets:
        file.write(f"DATASET: {dataset}\n" )
        for model in models:
            file.write(f"MODEL: {model}\n" )
            for alg in clustering_alg:

                print(f"Training and testing model {model} on dataset {dataset} and on task {task} with clustering algorithm {alg}...")

                modified_command = f"{train_command} --model={model} --dataset={dataset} --task={task} --clustering_algorithm={alg}"
                if dataset == "pubmed":
                    modified_command = f"{modified_command}  --kcore=True"

                output = subprocess.check_output(modified_command, shell=True) 
                output = output.decode()  

                print(f"Model trained and tested for clustering algorithm {alg}.")
                
                write_output(output,alg)

file.close()    

print("Sensitivity analysis on node clustering task completed. Results saved in", output_file)
