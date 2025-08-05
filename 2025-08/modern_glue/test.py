import google.generativeai as genai
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef, accuracy_score
import json
import os
import re

def parse_model_response(response_text):
    """
    Extracts the JSON object from the model's response, even if it's
    wrapped in markdown backticks or other text.
    """
    # Regex to find a JSON object enclosed in ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # If not in a markdown block, assume the whole text is the JSON
        # or it's a raw JSON object.
        json_str = response_text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the model's response.")
        return None

def main():
    """
    Main function to run the CoLA evaluation in a single shot.
    """
    # --- 1. Configuration ---
    print("Configuring the model...")
    try:
        # It's best practice to use an environment variable for your key
        # os.environ['GEMINI_API_KEY'] = 'YOUR_API_KEY'
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    except Exception:
        print("🚨 Error: GEMINI_API_KEY environment variable not set.")
        return
        
    model = genai.GenerativeModel('gemini-2.5-pro')

    # --- 2. Load and Prepare Data ---
    print("Loading CoLA validation dataset...")
    dataset = load_dataset('glue', 'cola', split='validation')
    ground_truth_labels = [example['label'] for example in dataset] # 0=unacceptable, 1=acceptable

    print(f"Formatting {len(dataset)} sentences for the prompt...")
    # Create a single string with all sentences numbered.
    sentences_text = "\n".join([f"{i+1}. {ex['sentence']}" for i, ex in enumerate(dataset)])

    # --- 3. Craft the "One-Shot" Prompt ---
    prompt = f"""
You are an expert English linguist. Your task is to evaluate a list of sentences for grammatical acceptability.
For each sentence in the list below, determine if it is "acceptable" or "unacceptable".

Your response MUST be a single JSON object. This object should contain one key, "results", which is a list of objects.
Each object in the "results" list must have two keys:
1. "index": The integer sentence number from the list.
2. "classification": Your judgment, which must be the string "acceptable" or "unacceptable".

Do not include any other text, explanations, or summaries in your response.

Here is the list of sentences:
---
{sentences_text}
---
"""

    print(prompt)
    # --- 4. Make the Single API Call ---
    print("\nSending the request to Gemini 2.5 Pro. This may take a moment...")
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        print(f"🚨 An error occurred during the API call: {e}")
        return

    # --- 5. Parse and Evaluate the Response ---
    print("Response received. Parsing and evaluating results...")
    parsed_data = parse_model_response(response.text)

    if not parsed_data or 'results' not in parsed_data:
        print("🚨 Error: Could not find 'results' in the parsed JSON data.")
        print("Full Model Output:\n---")
        print(response.text)
        print("---\n")
        return

    # Map text classifications to integer labels (0 or 1)
    label_map = {"unacceptable": 0, "acceptable": 1}
    
    # Sort results by index to ensure order matches ground truth, just in case
    results = sorted(parsed_data['results'], key=lambda x: x['index'])
    
    model_predictions = [label_map.get(item['classification']) for item in results]

    # Check for parsing errors or missing predictions
    if len(model_predictions) != len(ground_truth_labels):
        print(f"🚨 Warning: Mismatch in length. Ground truth: {len(ground_truth_labels)}, Predictions: {len(model_predictions)}")
        return
    if None in model_predictions:
        print("🚨 Error: Some classifications were not 'acceptable' or 'unacceptable'. Cannot score.")
        return

    # --- 6. Display Results ---
    mcc_score = matthews_corrcoef(ground_truth_labels, model_predictions)
    accuracy = accuracy_score(ground_truth_labels, model_predictions)

    print("\n--- ✅ Evaluation Complete ---")
    print(f"Matthews Correlation Coefficient (MCC): {mcc_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("---------------------------------")
    print("(For reference, the original BERT paper reported an MCC of 0.521 on CoLA.)")

# Matthews Correlation Coefficient (MCC): 0.6689
# Accuracy: 0.8619


if __name__ == '__main__':
    main()