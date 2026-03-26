import os
import re
import shutil
import subprocess

CONFIG_FILE = "config.yaml"

def update_config(updates):
    """
    Lit le fichier config.yaml et effectue un remplacement Regex des valeurs
    pour preserver les commentaires.
    `updates` est un dict: {'pnp_method': "'pgd'", 'gamma_pgd': '20.0', ...}
    """
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    for key, value in updates.items():
        # Cherche la clef exacte suivie de " :" ou ":" et remplace sa valeur jusqu'a la fin ou un commentaire inline
        pattern = rf"^({key}\s*:\s*)([^#\n]*)(\s*#.*)?$"
        # On remplace par la base $1, la nouvelle valeur, puis le commentaire s'il existe
        replacement = rf"\g<1>{value}\g<3>"
        
        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        else:
            # Si config pas trouvee (ex: add_observation_noise n'est pas encore dans le fichier)
            content += f"\n{key}: {value}"

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(content)


def run_experiment(exp_name, pnp_method, gamma_pgd=1.0, add_noise=False, iter_num=100):
    print(f"\n" + "="*60)
    print(f"🚀 RUNNING EXPERIMENT : {exp_name}")
    print(f"============================================================")
    
    # Prepares the config
    updates = {
        "name_folder_result": f"results_{exp_name}",
        "output_csv": f"metrics_{exp_name}.csv",
        "pnp_method": f"'{pnp_method}'",
        "gamma_pgd": str(gamma_pgd),
        "iter_num": str(iter_num),
        "add_observation_noise": "True" if add_noise else "False",
        "observation_noise_std": "0.1"
    }
    
    update_config(updates)
    
    # Executes the main pipeline
    # subprocess.call blocke jusqu'à la fin de l'execution
    subprocess.call(["python", "main.py"])
    print(f"✅ EXPERIMENT {exp_name} FINISHED.")


if __name__ == "__main__":
    # --- EXPERIMENT 1 : Ablation du Gamma (PGD vs HQS) ---
    # Cette experience montre l'evolution de la qualité de la "couture" et la fidélité
    gammas_to_test = [1.0, 10.0, 20.0, 50.0]
    
    # 1.1 HQS Baseline
    run_experiment(exp_name="EXP1_HQS_baseline", pnp_method="hqs")
    
    # 1.2 Differents Gammas pour PGD
    for g in gammas_to_test:
        run_experiment(exp_name=f"EXP1_PGD_gamma_{int(g)}", pnp_method="pgd", gamma_pgd=g)
        
    # --- EXPERIMENT 2 : Robustesse au bruit d'observation ---
    # On rajoute un bruit Gaussien (sigma=0.1) sur la zone visible pour voir 
    # si PGD peut "désapprendre" ce bruit alors que HQS va être contraint de le garder.
    run_experiment(exp_name="EXP2_HQS_noisy", pnp_method="hqs", add_noise=True)
    run_experiment(exp_name="EXP2_PGD_noisy_gamma_20", pnp_method="pgd", gamma_pgd=20.0, add_noise=True)

    # --- EXPERIMENT 3 : Impact de la longueur d'echantillonnage (NFE) ---
    # On descend a iter_num = 20 au lieu de 100
    run_experiment(exp_name="EXP3_HQS_fast_iter20", pnp_method="hqs", iter_num=20)
    run_experiment(exp_name="EXP3_PGD_fast_iter20_gamma_20", pnp_method="pgd", gamma_pgd=20.0, iter_num=20)

    print("\n🎉 ALL EXPERIMENTS COMPLETED SUCESSFULLY ! 🎉")
    print("Vous pouvez analyser les différents fichiers 'metrics_EXP*.csv' dans leurs dossiers 'results' respectifs.")
