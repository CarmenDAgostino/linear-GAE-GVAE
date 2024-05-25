import subprocess
import re
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# codice che verifica il comportamento dei modelli per il task di anomaly detection

# variabili
output_file = "results/ad_results"
train_command = "python train_ad.py"

# parametri
models = ["linear_ae","gcn_ae","gcn_vae","linear_vae"]#,"gcn_vae","linear_ae","linear_vae"
datasets = ["AIDS","ENZYMES","IMDB-BINARY"]#,"ENZYMES","IMDB-BINARY","REDDIT-MULTI-5K"


# funzione per scrivere il risultato
def write_output(output) :

    class_pattern = re.compile(r'Run for class (\d+).*?AUC score: (.*?)\nF1 score: (.*?)\nMean anomaly score: (\d+\.\d+)', re.DOTALL)
    mean_pattern = re.compile(
        r'Mean AUC score:\s*([\d.]+)\s*'
        r'Std of AUC scores:\s*([\d.]+)\s*'
        r'Mean F1 score:\s*([\d.]+)\s*'
        r'Std of F1 scores:\s*([\d.]+)\s*'
        r'Mean Anomaly Detection score:\s*([\d.]+)\s*'
        r'Std of AD scores:\s*([\d.]+)\s*'
        r'Total Running times\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*'
        r'Mean total running time:\s*([\d.]+)\s*'
        r'Std of total running time:\s*([\d.]+)',
        re.DOTALL
    )

    
    class_matches = class_pattern.findall(output)
    mean_match = mean_pattern.search(output)

    for match in class_matches:
        class_number, auc_score, f1_score, mean_anomaly_score = match
        file.write(f"Run for class {class_number}\n")
        file.write(f"AUC score: {auc_score}")
        file.write(f"F1 score: {f1_score}")
        file.write(f"Mean anomaly score: {mean_anomaly_score}\n\n")

    if mean_match != None:
        mean_values = mean_match.groups()
        file.write(f"Mean AUC score:  {mean_values[0]}\n")
        file.write(f"Mean F1 score:  {mean_values[2]}\n")
        file.write(f"Mean Anomale Detection score: {mean_values[4]}\n")
        file.write(f"Mean total running time:  {mean_values[7]}\n\n")


# training sui diversi dataset
for dataset in datasets:
    
    output_file_new = output_file + f"_{dataset}.txt"
    file = open(output_file_new, "w")

    file.write(f"DATASET: {dataset}\n\n" )

    for model in models:
        file.write(f"MODEL: {model}\n" )
        print(f"Training and testing model {model} on dataset {dataset} ...")

        modified_command = f"{train_command} --model={model} --dataset={dataset}"

        output = subprocess.check_output(modified_command, shell=True) 
        output = output.decode()  

        print(f"Model {model} trained and tested for dataset {dataset}. Result saved in {output_file_new}")

        write_output(output)
    file.close()

print("Test on anomaly detection task completed.")
