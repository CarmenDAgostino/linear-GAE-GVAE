import subprocess
import re
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# codice che verifica il comportamento dei modelli per il task di anomaly detection

# variabili
output_file = "results/ad_results.txt"
train_command = "python train_ad.py"

# parametri
models = ["gcn_ae","gcn_vae","linear_ae","linear_vae"]
datasets = ["AIDS","DD","ENZYMES","IMDB-BINARY","REDDIT-MULTI-5K"]


# funzione per scrivere il risultato
def write_output(output) :

    class_pattern = re.compile(r'Run for class (\d+).*?AUC score: (.*?)\nF1 score: (.*?)\nMean anomaly score: (.*?)', re.DOTALL)
    mean_pattern = re.compile(r'Mean AUC score:  (.*?)\nStd of AUC scores:  (.*?)\n\nMean F1 score:  (.*?)\nStd of AP scores:  (.*?)\n\nMean Anomale Detection score:  (.*?)\nStd of AP scores:  (.*?)\n\nTotal Running times\n \[(.*?)\]\nMean total running time:  (.*?)\nStd of total running time:  (.*?)', re.DOTALL) 

    class_matches = class_pattern.findall(output)
    mean_match = mean_pattern.search(output)

    for match in class_matches:
        class_number, auc_score, f1_score, mean_anomaly_score = match
        file.write(f"Run for class {class_number}\n")
        file.write(f"AUC score: {auc_score}")
        file.write(f"F1 score: {f1_score}")
        file.write(f"Mean anomaly score: {mean_anomaly_score}\n")

    if mean_match:
        mean_values = mean_match.groups()
        file.write(f"Mean AUC score:  {mean_values[0]}\n")
        file.write(f"Mean F1 score:  {mean_values[2]}\n")
        file.write(f"Mean Anomale Detection score {mean_values[4]}\n")
        file.write(f"Mean total running time:  {mean_values[7]}\n")


file = open(output_file, "a")

# training per diversi algoritmi di clustering
for dataset in datasets:
    file.write(f"DATASET: {dataset}\n\n" )

    for model in models:
        file.write(f"MODEL: {model}\n" )
        print(f"Training and testing model {model} on dataset {dataset} ...")

        modified_command = f"{train_command} --model={model} --dataset={dataset}"

        output = subprocess.check_output(modified_command, shell=True) 
        output = output.decode()  

        print(f"Model {model} trained and tested for dataset {dataset}.")

        write_output(output)


file.close()    

print("Test on anomaly detection task completed. Results saved in", output_file)
