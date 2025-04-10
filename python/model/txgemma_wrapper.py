import logging
import re # Import regex module for parsing
from typing import Optional, Dict, Any, Tuple

# Import transformers only if needed, to avoid making it a hard dependency if not used
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM # Or other relevant AutoModel class
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Define dummy classes if transformers is not installed
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name_or_path: str, **kwargs):
            raise ImportError("transformers library not installed. Cannot load tokenizer.")
    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_name_or_path: str, **kwargs):
            raise ImportError("transformers library not installed. Cannot load model.")
    class torch:
        dtype = None
        bfloat16 = None
        @staticmethod
        def cuda_is_available(): return False
        @staticmethod
        def device(type): return type

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Loading ---

def load_txgemma_model(model_name_or_path: str) -> Optional[Tuple[Any, Any]]:
    """Loads a TxGemma model and tokenizer from Hugging Face.

    Args:
        model_name_or_path: The name (on Hugging Face Hub) or local path of the model.

    Returns:
        A tuple containing the loaded model and tokenizer, or None if loading fails.
    """
    if not TRANSFORMERS_AVAILABLE:
        logging.error("Transformers library not installed. Cannot load model.")
        return None

    try:
        logging.info(f"Loading tokenizer for {model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        logging.info(f"Loading model {model_name_or_path}...")

        # Add device map and quantization if needed for large models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # --- Model loading configuration --- #
        # Consider quantization for larger models if needed/supported
        # from transformers import BitsAndBytesConfig
        # quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            # quantization_config=quantization_config, # Example quantization
            torch_dtype=torch.bfloat16, # Use bfloat16 if possible
            device_map=device
        )

        logging.info(f"Successfully loaded model and tokenizer for {model_name_or_path}.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model {model_name_or_path}: {e}", exc_info=True)
        return None

# --- Input Preparation ---

def prepare_input_for_txgemma(prompt_text: str, tokenizer: Any) -> Optional[Dict[str, Any]]:
    """Tokenizes the input prompt text for the TxGemma model.

    Args:
        prompt_text: The textual summary prompt generated by the feature extractor.
        tokenizer: The loaded Hugging Face tokenizer.

    Returns:
        A dictionary containing tokenized inputs ready for model.generate(), or None.
    """
    logging.info("Preparing input for TxGemma by tokenizing prompt...")
    if not prompt_text:
        logging.error("Cannot prepare input: Prompt text is empty or None.")
        return None
    if tokenizer is None:
        logging.error("Cannot prepare input: Tokenizer is None.")
        return None

    try:
        # Tokenize the prompt
        inputs = tokenizer(prompt_text, return_tensors="pt") # No padding/truncation by default, model handles context

        # Move inputs to the same device as the model
        if hasattr(tokenizer, 'model_device'):
             inputs = inputs.to(tokenizer.model_device)
        elif torch.cuda.is_available():
             inputs = inputs.to('cuda')

        logging.info("Input prompt tokenization successful.")
        return inputs

    except Exception as e:
        logging.error(f"Error during input tokenization: {e}", exc_info=True)
        return None

# --- Prediction Generation ---

def generate_noael_prediction(model: Any, tokenizer: Any, inputs: Dict[str, Any]) -> Optional[str]:
    """Generates predictions using the TxGemma model.

    Args:
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
        inputs: The tokenized input dictionary from prepare_input_for_txgemma.

    Returns:
        The raw text output from the model, or None if generation fails.
    """
    logging.info("Generating NOAEL prediction...")
    if model is None or tokenizer is None or inputs is None:
        logging.error("Cannot generate prediction: Model, tokenizer, or inputs are None.")
        return None

    try:
        # Set generation parameters (adjust max_new_tokens as needed)
        generation_params = {
            "max_new_tokens": 150, # Allow more tokens for NOAEL value, units, and maybe brief rationale
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
             # Consider adding parameters like temperature, top_k/top_p if needed
            "do_sample": False, # Typically False for factual extraction like NOAEL
        }

        input_ids = inputs["input_ids"].to(model.device)
        # Attention mask might not be needed if no padding was applied, but include if present
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
            generation_params["attention_mask"] = attention_mask

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, **generation_params)

        # Decode the full output first, then isolate the generated part
        full_output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Input prompt text was part of the input_ids, find its length after decoding
        # (Decoding might slightly alter length, safer to decode generated part only)
        generated_part = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

        logging.info(f"Raw model output (generated part): {generated_part}")
        return generated_part.strip()

    except Exception as e:
        logging.error(f"Error during model generation: {e}", exc_info=True)
        return None

# --- Output Processing ---

def process_txgemma_output(raw_output: str) -> Optional[Dict[str, Any]]:
    """Parses the raw text output from TxGemma to extract NOAEL value and units.

    Uses regex to find patterns like 'NOAEL: 100 mg/kg' or 'NOAEL = 50 mg/kg/day'.

    Args:
        raw_output: The raw text string generated by the model.

    Returns:
        A dictionary containing the structured prediction results (e.g., {'noael': 100, 'units': 'mg/kg'}),
        or a dictionary with just the raw_output if parsing fails.
    """
    logging.info("Processing TxGemma output...")
    if not raw_output:
        logging.error("Cannot process output: Raw output is empty or None.")
        return {"raw_output": "", "predicted_noael": None, "units": None}

    # Regex to capture NOAEL value and units. Handles variations.
    # Looks for "NOAEL" followed by optional separator (:, =), then number, then units.
    # Units can include mg/kg, mg/kg/day, ppm, etc.
    # Makes units part flexible: (\s*([\w\/\-%]+(?:\/[\w\-%]+)*))
    # - \s* : optional space
    # - ([\w\/\-%]+) : captures word characters, /, -, % (e.g., mg/kg, ppm, %)
    # - (?:\/[\w\-%]+)* : optionally captures more slashes and terms (e.g., /day)
    patterns = [
        r"NOAEL\s*[:=]?\s*(\d+\.?\d*)\s*([\w\/\-%]+(?:\/[\w\-%]+)*)", # Standard NOAEL: 100 mg/kg/day
        r"(\d+\.?\d*)\s*([\w\/\-%]+(?:\/[\w\-%]+)*)\s+as\s+NOAEL",     # 100 mg/kg as NOAEL
        r"predicted\s+NOAEL\s*[:=]?\s*(\d+\.?\d*)\s*([\w\/\-%]+(?:\/[\w\-%]+)*)" # predicted NOAEL: 100 mg/kg
    ]

    extracted_noael = None
    extracted_units = None

    for pattern in patterns:
        match = re.search(pattern, raw_output, re.IGNORECASE)
        if match:
            try:
                extracted_noael = float(match.group(1))
                extracted_units = match.group(2).strip()
                logging.info(f"Parsed NOAEL: {extracted_noael}, Units: {extracted_units}")
                break # Stop after first successful match
            except (IndexError, ValueError) as e:
                logging.warning(f"Regex matched pattern '{pattern}' but failed to extract groups: {e}")
                continue # Try next pattern

    result = {
        "raw_output": raw_output,
        "predicted_noael": extracted_noael,
        "units": extracted_units
    }

    if extracted_noael is None:
        logging.warning(f"Could not parse NOAEL value and units from raw output: '{raw_output}'")

    return result

# --- Orchestration ---

def predict_with_txgemma(prompt_text: str, model_name_or_path: str) -> Optional[Dict[str, Any]]:
    """Full pipeline: Load model, prepare input (tokenize prompt), predict, process output.

    Args:
        prompt_text: The prompt string generated by the feature extractor.
        model_name_or_path: The name or path of the TxGemma model.

    Returns:
        A dictionary with the structured prediction result, or None if any step fails.
    """
    logging.info(f"Starting TxGemma prediction pipeline for model: {model_name_or_path}")

    # --- SANITY CHECK REMOVED --- #
    # sanity_check_prompt = "Briefly describe the primary mechanism of action of Aspirin."
    # logging.warning(f"--- RUNNING SANITY CHECK --- Using fixed prompt: '{sanity_check_prompt}'")
    # prompt_to_use = sanity_check_prompt
    prompt_to_use = prompt_text # Use the actual prompt
    # --- End Sanity Check --- #

    # 1. Load Model
    model_load_result = load_txgemma_model(model_name_or_path)
    if model_load_result is None:
        return None
    model, tokenizer = model_load_result

    # Associate tokenizer with model's device
    tokenizer.model_device = model.device

    # 2. Prepare Input (Tokenize Prompt)
    inputs = prepare_input_for_txgemma(prompt_to_use, tokenizer)
    if inputs is None:
        return None

    # 3. Generate Prediction
    raw_output = generate_noael_prediction(model, tokenizer, inputs)
    if raw_output is None:
        return {"raw_output": "Model generation failed.", "predicted_noael": None, "units": None}

    # 4. Process Output (Parse Raw Text)
    structured_result = process_txgemma_output(raw_output)

    logging.info("TxGemma prediction pipeline finished.")
    return structured_result


# Example Usage (Optional - for testing)
# if __name__ == '__main__':
#     if not TRANSFORMERS_AVAILABLE:
#         print("Transformers library not installed. Cannot run example.")
#     else:
#         # --- Create Dummy Prompt --- #
#         dummy_prompt = ("Study Design Summary:\n" 
#                       "- Study ID: TEST001\n" 
#                       "- Species: Rat\n" 
#                       "- Route of Administration: Oral\n\n" 
#                       "Dosing Information:\n" 
#                       "- Treatment: Compound X, Dose: 0 mg/kg\n" 
#                       "- Treatment: Compound X, Dose: 10 mg/kg\n" 
#                       "- Treatment: Compound X, Dose: 50 mg/kg\n" 
#                       "- Treatment: Compound X, Dose: 200 mg/kg\n\n" 
#                       "Based on the SEND study data summarized above, predict the No Observed Adverse Effect Level (NOAEL). Provide the value and units.")
#         print("--- Dummy Prompt ---")
#         print(dummy_prompt)
#
#         # --- Specify Model --- #
#         # Replace with the actual TxGemma model name
#         test_model_name = "google/txgemma-2b-predict" # Or "gpt2" for basic structure test
#         print(f"\n--- Using model: {test_model_name} for testing --- ")
#
#         # --- Run Pipeline --- #
#         try:
#             prediction = predict_with_txgemma(dummy_prompt, test_model_name)
#             print("\n--- Prediction Result ---")
#             if prediction is not None:
#                 print(prediction)
#             else:
#                 print("Prediction failed.")
#         except ImportError as e:
#              print(f"Error: {e}. Make sure transformers and torch are installed.")
#         except Exception as e:
#              print(f"An unexpected error occurred: {e}") 