import pandas as pd
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_noael_from_prediction(prediction_result: Dict[str, Any], study_info: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Determines the final NOAEL value based on parsed model predictions.

    Uses the 'predicted_noael' and 'units' fields parsed by the wrapper.
    Further refinement could involve dose group analysis if needed.

    Args:
        prediction_result: The structured output from the model wrapper
                           (e.g., output of txgemma_wrapper.process_txgemma_output,
                            should contain 'predicted_noael' and 'units' keys).
        study_info: Optional dictionary containing study design info (e.g., from TS domain)

    Returns:
        A dictionary containing the calculated NOAEL value, units, rationale, etc.,
        or None if calculation is not possible.
    """
    logging.info("Calculating NOAEL from parsed prediction...")
    if not prediction_result:
        logging.error("Cannot calculate NOAEL: Prediction result is empty or None.")
        return None

    parsed_noael = prediction_result.get('predicted_noael')
    parsed_units = prediction_result.get('units')
    raw_output = prediction_result.get('raw_output', '')

    if parsed_noael is not None and parsed_units is not None:
        final_noael = {
            "value": parsed_noael,
            "units": parsed_units,
            "determination_method": "Parsed from TxGemma prediction",
            "raw_model_output": raw_output # Include raw output for reference
        }
        logging.info(f"Determined final NOAEL: {final_noael['value']} {final_noael['units']}")
        return final_noael
    else:
        logging.warning(f"Could not determine NOAEL from parsed prediction. Raw output: '{raw_output[:100]}...'")
        # Return structure indicating failure but include raw output
        return {
            "value": None,
            "units": None,
            "determination_method": "Parsing Failed",
            "raw_model_output": raw_output
        }

# Example Usage (Optional)
# if __name__ == '__main__':
#     # Example 1: Successful parse
#     parsed_prediction_success = {'raw_output': 'The NOAEL is estimated to be 100 mg/kg/day.', 'predicted_noael': 100.0, 'units': 'mg/kg/day'}
#     noael_info_success = calculate_noael_from_prediction(parsed_prediction_success)
#     print("\n--- Calculated NOAEL Info (Success Example) ---")
#     print(noael_info_success)
#
#     # Example 2: Parsing failed
#     parsed_prediction_fail = {'raw_output': 'Adverse effects were seen at all doses.', 'predicted_noael': None, 'units': None}
#     noael_info_fail = calculate_noael_from_prediction(parsed_prediction_fail)
#     print("\n--- Calculated NOAEL Info (Fail Example) ---")
#     print(noael_info_fail) 