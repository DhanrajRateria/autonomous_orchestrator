import yaml
import subprocess
import json
import pandas as pd
import os
import shutil
from pathlib import Path
from datetime import datetime
import logging
import time

# Configure basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("experiment_runner")

# --- Configuration ---
BASE_CONFIG_FILE = "he_config.yaml"  # Your working HE config (will be copied and modified)
CANCER_DETECTION_SCRIPT = Path("federated_learning_framework/examples/cancer_detection.py")
RESULTS_DIR = Path("experiment_results")
NUM_CLIENTS_FOR_EXPERIMENTS = 10 # Consistent number of clients for comparison
NUM_RUNS_PER_EXPERIMENT = 1 # Number of times to repeat each experiment for averaging (set to >1 for robustness)


# --- Experiment Definitions ---
# Each dictionary defines a named experiment and the modifications to the base config
EXPERIMENTS = [
    {
        "name": "S1_Normal_FL_HE",
        "description": "Standard Federated Learning with Homomorphic Encryption. No quality weighting, no bad clients.",
        "config_overrides": {
            "quality": {
                "enabled": False,
            },
            "robustness": {
                "fraction_noisy_clients": 0.0,
                "fraction_adversarial_clients": 0.0,
            }
        }
    },
    {
        "name": "S2_FL_HE_With_Bad_Clients_FedAvg",
        "description": "FL+HE with noisy and adversarial clients, using standard FedAvg (quality weighting disabled).",
        "config_overrides": {
            "quality": {
                "enabled": False,
            },
            "robustness": {
                "fraction_noisy_clients": 0.2,
                "label_flip_probability": 0.15,
                "fraction_adversarial_clients": 0.1,
                "attack_scale_factor": -2.0,
            }
        }
    },
    {
        "name": "S3a_FL_HE_Bad_Clients_QualityRaw",
        "description": "FL+HE with bad clients, quality-aware aggregation (raw scores, no clipping).",
        "config_overrides": {
            "quality": {
                "enabled": True,
                "score_epsilon": 1e-6,
                "robust_score_aggregation": False, # RAW scores used
            },
            "robustness": {
                "fraction_noisy_clients": 0.2,
                "label_flip_probability": 0.15,
                "fraction_adversarial_clients": 0.1,
                "attack_scale_factor": -2.0,
            }
        }
    },
    {
        "name": "S3b_FL_HE_Bad_Clients_QualityRobust_5_95",
        "description": "FL+HE with bad clients, quality-aware aggregation with robust percentile clipping (5th-95th).",
        "config_overrides": {
            "quality": {
                "enabled": True,
                "score_epsilon": 1e-6,
                "robust_score_aggregation": True,
                "score_clip_percentile_lower": 5.0,
                "score_clip_percentile_upper": 95.0,
            },
            "robustness": {
                "fraction_noisy_clients": 0.2,
                "label_flip_probability": 0.15,
                "fraction_adversarial_clients": 0.1,
                "attack_scale_factor": -2.0,
            }
        }
    },
    {
        "name": "S4_FL_HE_Bad_Clients_QualityRobust_10_90",
        "description": "FL+HE with bad clients, quality-aware aggregation with tighter robust clipping (10th-90th).",
        "config_overrides": {
            "quality": {
                "enabled": True,
                "score_epsilon": 1e-6,
                "robust_score_aggregation": True,
                "score_clip_percentile_lower": 10.0,
                "score_clip_percentile_upper": 90.0,
            },
            "robustness": {
                "fraction_noisy_clients": 0.2, # Keep consistent for comparing clipping
                "label_flip_probability": 0.15,
                "fraction_adversarial_clients": 0.1,
                "attack_scale_factor": -2.0,
            }
        }
    },
    # Add more experiments here if needed (e.g., different fractions of bad clients)
]

def deep_update(source, overrides):
    """
    Recursively update a dictionary.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            source[key] = deep_update(source[key], value)
        else:
            source[key] = value
    return source

def run_single_experiment(exp_name: str, config_overrides: dict, run_number: int, base_config_path: str):
    """
    Sets up and runs a single experiment instance.
    Returns the path to the history file and the result_dir for this run.
    """
    logger.info(f"--- Starting Experiment: {exp_name} (Run {run_number}) ---")

    # Load base configuration
    try:
        with open(base_config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load base config {base_config_path}: {e}")
        return None, None

    # Apply overrides
    config_data = deep_update(config_data, config_overrides)
    
    # Create a unique directory for this specific experiment run's artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_experiment_run_dir = RESULTS_DIR / f"{exp_name}_run{run_number}_{timestamp}"
    current_experiment_run_dir.mkdir(parents=True, exist_ok=True)

    # Modify system paths in the config to point to this unique directory
    project_name_for_run = f"{config_data.get('project_name', 'fl_project')}_{exp_name}_run{run_number}"
    config_data['project_name'] = project_name_for_run # To make history file unique
    config_data['system']['checkpoint_dir'] = str(current_experiment_run_dir / "checkpoints")
    config_data['system']['result_dir'] = str(current_experiment_run_dir / "results") # Main output, history.json here

    # Save the modified config for this run
    temp_config_path = current_experiment_run_dir / f"config_for_run.yaml"
    try:
        with open(temp_config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved run-specific config to {temp_config_path}")
    except Exception as e:
        logger.error(f"Failed to save run-specific config: {e}")
        return None, None

    # Construct command
    # Assuming cancer_detection.py is in the examples subdir relative to where this script is
    # Adjust path if needed
    script_to_run = CANCER_DETECTION_SCRIPT 
    if not script_to_run.exists():
        # Try another common relative path if running from root of framework
        script_to_run = Path("examples") / "cancer_detection.py"
        if not script_to_run.exists():
             logger.error(f"Cannot find cancer_detection.py at {CANCER_DETECTION_SCRIPT} or {script_to_run}")
             return None, None


    command = [
        "python", str(script_to_run),
        "--config", str(temp_config_path),
        "--clients", str(NUM_CLIENTS_FOR_EXPERIMENTS)
        # Add other necessary args for cancer_detection.py if they aren't in config
        # e.g., --data if it's not consistently in base_config_file
    ]
    logger.info(f"Executing command: {' '.join(command)}")

    # Run the FL script
    # The log file for the FL run itself will be created by its own setup_logging
    # inside the current_experiment_run_dir / "results" / "run_logs"
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"Experiment {exp_name} (Run {run_number}) failed with return code {process.returncode}")
            logger.error("STDOUT:")
            logger.error(stdout)
            logger.error("STDERR:")
            logger.error(stderr)
            # Save stdout/stderr to files in the experiment run dir for inspection
            with open(current_experiment_run_dir / "run.stdout", "w") as f_out: f_out.write(stdout)
            with open(current_experiment_run_dir / "run.stderr", "w") as f_err: f_err.write(stderr)
            return None, str(current_experiment_run_dir / "results") # Return result dir even on failure for logs
        else:
            logger.info(f"Experiment {exp_name} (Run {run_number}) completed successfully.")
            with open(current_experiment_run_dir / "run.stdout", "w") as f_out: f_out.write(stdout) # Save for reference
            if stderr:
                 with open(current_experiment_run_dir / "run.stderr", "w") as f_err: f_err.write(stderr)


    except Exception as e:
        logger.error(f"Exception during experiment {exp_name} (Run {run_number}): {e}", exc_info=True)
        return None, str(current_experiment_run_dir / "results")

    # Path to the history file generated by cancer_detection.py
    # It should be inside config_data['system']['result_dir'] / <project_name_for_run>_training_history.json
    history_file_path = Path(config_data['system']['result_dir']) / f"{project_name_for_run}_training_history.json"
    if not history_file_path.exists():
        logger.error(f"History file not found at {history_file_path} for {exp_name} (Run {run_number})")
        return None, str(Path(config_data['system']['result_dir']))
        
    return str(history_file_path), str(Path(config_data['system']['result_dir']))


def extract_metrics_from_history(history_file_path: str):
    """
    Extracts key metrics from a single experiment's history file.
    Focuses on final test metrics and metrics from the last training round.
    """
    if not history_file_path or not Path(history_file_path).exists():
        return {
            "final_test_accuracy": None, "final_test_loss": None, "final_test_f1": None,
            "final_test_precision": None, "final_test_recall": None, "final_test_auc": None,
            "last_val_accuracy": None, "last_val_loss": None, "total_rounds_completed": 0
        }

    with open(history_file_path, 'r') as f:
        history_data = json.load(f)

    results = {}
    training_log = history_data.get("training_log", [])

    # Final Test Metrics
    # The 'test_metrics' should be at the top level of history_data or in the last round's entry
    final_test_metrics = history_data.get("final_global_metrics") # Assuming cancer_detection.py stores test metrics here in latest version
    if not final_test_metrics and training_log and "test_metrics" in training_log[-1]: # Fallback to older structure
        final_test_metrics = training_log[-1].get("test_metrics")

    if final_test_metrics:
        results["final_test_accuracy"] = final_test_metrics.get("accuracy")
        results["final_test_loss"] = final_test_metrics.get("loss")
        results["final_test_f1"] = final_test_metrics.get("f1_score") # or f1_score_macro
        results["final_test_precision"] = final_test_metrics.get("precision") # or precision_macro
        results["final_test_recall"] = final_test_metrics.get("recall") # or recall_macro
        results["final_test_auc"] = final_test_metrics.get("auc") # or auc_macro_ovr
    else:
        results.update({
            "final_test_accuracy": None, "final_test_loss": None, "final_test_f1": None,
            "final_test_precision": None, "final_test_recall": None, "final_test_auc": None
        })

    # Metrics from the last completed training round (validation)
    if training_log:
        last_round_data = training_log[-1]
        results["total_rounds_completed"] = last_round_data.get("round", 0)
        last_val_metrics = last_round_data.get("global_metrics", {})
        results["last_val_accuracy"] = last_val_metrics.get("accuracy")
        results["last_val_loss"] = last_val_metrics.get("loss")
    else:
        results["total_rounds_completed"] = 0
        results["last_val_accuracy"] = None
        results["last_val_loss"] = None
        
    results["history_file"] = str(history_file_path)
    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_experiment_results = []
    start_all_experiments_time = time.time()

    if not Path(BASE_CONFIG_FILE).exists():
        logger.error(f"Base configuration file '{BASE_CONFIG_FILE}' not found. Please create it.")
        return

    for experiment in EXPERIMENTS:
        exp_name = experiment["name"]
        config_overrides = experiment["config_overrides"]
        logger.info(f"===== Starting Experiment Group: {exp_name} =====")
        logger.info(f"Description: {experiment.get('description', 'N/A')}")

        run_metrics_for_experiment = []
        for run_idx in range(1, NUM_RUNS_PER_EXPERIMENT + 1):
            history_file, result_dir_for_run = run_single_experiment(exp_name, config_overrides, run_idx, BASE_CONFIG_FILE)
            
            if history_file:
                metrics = extract_metrics_from_history(history_file)
                metrics["experiment_name"] = exp_name
                metrics["run_number"] = run_idx
                metrics["result_dir"] = result_dir_for_run
                all_experiment_results.append(metrics)
                run_metrics_for_experiment.append(metrics)
            else:
                logger.warning(f"Skipping metrics extraction for {exp_name} Run {run_idx} due to run failure or missing history.")
                # Add a placeholder for failed runs if desired
                all_experiment_results.append({
                    "experiment_name": exp_name, "run_number": run_idx, "status": "failed",
                    "result_dir": result_dir_for_run, "history_file": None
                })

        # Optional: Log average metrics for this experiment group if NUM_RUNS_PER_EXPERIMENT > 1
        if NUM_RUNS_PER_EXPERIMENT > 1 and run_metrics_for_experiment:
            df_exp_runs = pd.DataFrame(run_metrics_for_experiment)
            logger.info(f"--- Average Metrics for {exp_name} over {len(run_metrics_for_experiment)} successful runs ---")
            # Calculate mean for numeric columns only, handling potential NaNs
            numeric_cols = df_exp_runs.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                if col not in ["run_number"]: # Don't average run_number
                     logger.info(f"Avg {col}: {df_exp_runs[col].mean():.4f} (Std: {df_exp_runs[col].std():.4f})")
            logger.info("---------------------------------------------------")


    # Save all results to a CSV file
    summary_file_path = RESULTS_DIR / "experiments_summary.csv"
    df_results = pd.DataFrame(all_experiment_results)
    df_results.to_csv(summary_file_path, index=False)
    logger.info(f"All experiment results saved to: {summary_file_path}")

    end_all_experiments_time = time.time()
    logger.info(f"Total time for all experiments: {(end_all_experiments_time - start_all_experiments_time)/60:.2f} minutes.")

if __name__ == "__main__":
    main()