#!/usr/bin/env python
import yaml
import json
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def enable_he_in_config(input_config_path, output_config_path):
    """Loads an existing config and enables HE settings."""

    try:
        with open(input_config_path, 'r') as f:
            # Use SafeLoader for security
            config = yaml.safe_load(f)
    except FileNotFoundError:
         logging.error(f"Input config file not found: {input_config_path}")
         return
    except yaml.YAMLError as e:
         logging.error(f"Error parsing input YAML file {input_config_path}: {e}")
         return
    except Exception as e:
         logging.error(f"An unexpected error occurred loading {input_config_path}: {e}")
         return


    # Modify the crypto section
    if 'crypto' not in config:
        config['crypto'] = {} # Create section if it doesn't exist

    config['crypto']['enabled'] = True
    # Keep existing scheme/params if they exist, otherwise use defaults
    config['crypto'].setdefault('scheme', 'CKKS')
    config['crypto'].setdefault('poly_modulus_degree', 8192)
    config['crypto'].setdefault('coeff_mod_bit_sizes', [40, 20, 40])
    config['crypto'].setdefault('global_scale', 2**40)
    config['crypto'].setdefault('security_level', 128)

    logging.info(f"Enabled Homomorphic Encryption in the configuration.")
    logging.info(f"  Scheme: {config['crypto']['scheme']}")
    logging.info(f"  Poly Modulus Degree: {config['crypto']['poly_modulus_degree']}")

    # Save the updated config
    try:
        output_path_obj = Path(output_config_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(output_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logging.info(f"Configuration with HE enabled saved to: {output_config_path}")
        print(f"\nSuccessfully created HE-enabled config: {output_config_path}")
        print(f"You can now run the framework using this config, e.g.:")
        print(f"python cancer_detection.py --config {output_config_path} --clients 5")

    except Exception as e:
        logging.error(f"Failed to save updated configuration to {output_config_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enable Homomorphic Encryption in an existing FL config file.")
    parser.add_argument("--input", required=True, help="Path to the input YAML configuration file.")
    parser.add_argument("--output", default="he_enabled_config.yaml", help="Path to save the HE-enabled output YAML configuration file.")

    args = parser.parse_args()

    enable_he_in_config(args.input, args.output)