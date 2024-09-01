import csv
import json


def read_csv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        return list(reader)


def find_best_parameters(data, models):
    best_params = {}
    for model in models:
        model_data = [row for row in data if row['model'] == model]
        if not model_data:
            continue

        best_row = max(model_data, key=lambda x: float(x['f1']))
        best_params[model] = {
            'optimizer': best_row['optimizer'],
            'hidden_layers': best_row['hidden_layers'],
            'learning_rate': float(best_row['learning_rate']),
            'epoch': int(best_row['epoch']),
            'f1_score': float(best_row['f1']),
            'accuracy': float(best_row['accuracy']),
            'val_loss': float(best_row['val_loss'])
        }

    return best_params


def main():
    file_path = 'outputs/wz_classifier_experiments/experiment_results_20240901_110458.csv'
    models = ["electra_small_discriminator", "distilbert", "t5", "roberta", "bert"]
    output_file = 'models/optimal_wz_classifier_validation_hyperparams.json'

    try:
        data = read_csv_file(file_path)
        best_params = find_best_parameters(data, models)

        with open(output_file, 'w') as f:
            json.dump(best_params, f, indent=2)

        print(f"Results have been saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()