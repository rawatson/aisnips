import google.generativeai as genai
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef, accuracy_score
import json
import os
import re
import time
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, List, Dict, Any

# ==============================================================================
# 1. UTILITY FUNCTIONS
# ==============================================================================

def load_cola_dataset() -> (List[Dict[str, Any]], List[int]):
    """Loads the CoLA validation set."""
    print("Loading CoLA validation dataset...")
    dataset = load_dataset('glue', 'cola', split='validation')
    ground_truth_labels = [ex['label'] for ex in dataset]
    return list(dataset), ground_truth_labels

def save_results_to_tsv(strategy_name: str, predictions: List[str]):
    """Saves a list of predictions to a single-column TSV file."""
    filename = f"{strategy_name}_predictions.tsv"
    with open(filename, 'w') as f:
        f.write("classification\n") # Header
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Results saved to {filename}")

def calculate_and_print_metrics(strategy_name: str, ground_truth: List[int], model_predictions: List[str]):
    """Calculates and prints MCC and Accuracy."""
    label_map = {"unacceptable": 0, "acceptable": 1}
    
    # Convert string predictions to integer labels
    int_predictions = [label_map.get(p.lower().strip(), -1) for p in model_predictions]
    
    if len(int_predictions) != len(ground_truth):
        print(f"🚨 Warning: Mismatch in length. Ground truth: {len(ground_truth)}, Predictions: {len(int_predictions)}")
        return
        
    mcc_score = matthews_corrcoef(ground_truth, int_predictions)
    accuracy = accuracy_score(ground_truth, int_predictions)

    print("\n--- ✅ Evaluation Complete ---")
    print(f"Strategy: {strategy_name}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("---------------------------------")

def parse_json_from_response(response_text: str) -> Dict:
    """Extracts a JSON object from a model's text response."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    json_str = match.group(1) if match else response_text
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from response.")
        return None

# ==============================================================================
# 2. STRATEGY IMPLEMENTATION FUNCTIONS
# ==============================================================================
# Each function implements a different way of getting predictions from the model.

def run_one_shot_json_strategy(strategy: 'Strategy', dataset: List[Dict]) -> List[str]:
    """
    Generates all predictions in a single API call, expecting a JSON response.
    """
    print(f"Formatting {len(dataset)} sentences for the one-shot prompt...")
    model = genai.GenerativeModel(strategy.model_name)
    sentences_text = "\n".join([f"{i+1}. {ex['sentence']}" for i, ex in enumerate(dataset)])
    prompt = strategy.prompt_template.format(sentences_text=sentences_text)

    print("Sending single, large request to model...")
    response = model.generate_content(prompt)
    
    print("Parsing JSON response...")
    parsed_data = parse_json_from_response(response.text)
    if not parsed_data or 'results' not in parsed_data:
        raise ValueError("Could not find 'results' in the parsed JSON data.")
    
    results = sorted(parsed_data['results'], key=lambda x: x['index'])
    return [item['classification'] for item in results]

def run_one_by_one_strategy(strategy: 'Strategy', dataset: List[Dict]) -> List[str]:
    """
    Generates predictions by making one API call for each sentence. Slower but simpler.
    """
    print(f"Iterating through {len(dataset)} sentences one-by-one...")
    model = genai.GenerativeModel(strategy.model_name)
    predictions = []
    
    for example in tqdm(dataset, desc="Processing sentences"):
        prompt = strategy.prompt_template.format(sentence=example['sentence'])
        try:
            response = model.generate_content(prompt)
            # Simple cleanup of the model's output
            prediction = response.text.strip().lower()
            predictions.append(prediction)
        except Exception as e:
            print(f"An error occurred on a sentence: {e}")
            predictions.append("error") # Add a placeholder for failed calls
        time.sleep(1) # Be kind to the API and avoid rate limits
        
    return predictions

# ==============================================================================
# 3. STRATEGY DEFINITION AND EXECUTION
# ==============================================================================

@dataclass
class Strategy:
    name: str
    model_name: str
    eval_function: Callable[['Strategy', List[Dict]], List[str]]
    prompt_template: str

def evaluate_strategy(strategy: Strategy, dataset: List[Dict], ground_truth: List[int]):
    """Orchestrates the evaluation for a single defined strategy."""
    print(f"\n{'='*20} Running Strategy: {strategy.name} {'='*20}")
    try:
        predictions = strategy.eval_function(strategy, dataset)
        save_results_to_tsv(strategy.name, predictions)
        calculate_and_print_metrics(strategy.name, ground_truth, predictions)
    except Exception as e:
        print(f"🚨 Strategy '{strategy.name}' failed to run: {e}")

def main():
    """Main entry point to define and run all evaluation strategies."""
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    except Exception:
        print("🚨 Error: Please set the GEMINI_API_KEY environment variable.")
        return

    # --- DEFINE ALL STRATEGIES TO TEST HERE ---
    
    one_shot_prompt = """
You are an expert English linguist. Your task is to evaluate a list of sentences for grammatical acceptability.
For each sentence in the list below, determine if it is "acceptable" or "unacceptable".
Your response MUST be a single JSON object with one key, "results", which is a list of objects.
Each object must have two keys: "index" (the sentence number) and "classification" ("acceptable" or "unacceptable").
Do not include any other text in your response.

Here is the list of sentences:
---
{sentences_text}
---
"""

    one_by_one_prompt = """
Is the following sentence grammatically acceptable or unacceptable?
Respond with only the single word "acceptable" or "unacceptable".

Sentence: "{sentence}"
Answer:"""

    strategies_to_run = [
        #Matthews Correlation Coefficient (MCC): 0.6689
        #Accuracy: 0.8619
        Strategy(
            name="gemini_2_5_pro_one_shot_json",
            model_name="gemini-2.5-pro",
            eval_function=run_one_shot_json_strategy,
            prompt_template=one_shot_prompt
        ),
        #Matthews Correlation Coefficient (MCC): 0.5935
        #Accuracy: 0.8236
        Strategy(
            name="gemini_2_5_flash_lite_one_shot_json",
            model_name="gemini-2.5-flash-lite",
            eval_function=run_one_shot_json_strategy,
            prompt_template=one_shot_prompt
        ),
    ]


    # --- RUN THE EVALUATION ---
    dataset, ground_truth_labels = load_cola_dataset()
    
    #for strategy in strategies_to_run:
    #    evaluate_strategy(strategy, dataset, ground_truth_labels)
    evaluate_strategy(strategies_to_run[1], dataset, ground_truth_labels)

if __name__ == '__main__':
    main()