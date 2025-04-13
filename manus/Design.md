## 1. Challenges with TxGemma redict models

The Initial strategy was to use TxGemma for direct NOAEL prediction, but encountered significant hurdles:

*   **Task Mismatch (Text Generation vs. Quantitative Regression):** The fundamental issue was using a model optimized for natural language understanding and generation for a quantitative regression task. `txgemma-2b-predict`, while knowledgeable in the biomedical domain, is not inherently designed to perform precise numerical calculations or regressions based on summarized text inputs. Its primary function is text prediction and generation.
*   **Inconsistent and Unparsable Output:** The model rarely produced the desired output format (e.g., a specific number and unit like "50 mg/kg/day"). Instead, outputs were often:
    *   **Non-numerical:** Single letters (e.g., "A"), generic text fragments.
    *   **Irrelevant but Contextual:** SMILES strings during sanity checks (indicating some context understanding but not task execution).
    *   **Unstructured:** Free text that might mention study details but lacked a clearly identifiable or parsable NOAEL value.
    This inconsistency made reliable extraction of a numerical result impossible, leading to frequent "Parsing Failed" states.
*   **Input Representation Difficulty:** Translating structured, tabular SEND data into a textual summary for the LLM loses some numerical precision and relational information inherent in the original tables. Asking the LLM to reason quantitatively based on this secondary textual representation is less direct and potentially more error-prone than using the structured data directly.
*   **Prompt Engineering Complexity:** LLM outputs are highly sensitive to input prompts. Crafting a prompt that could reliably instruct the model to not only understand the summarized study but also perform the specific calculation and return it in the desired format proved extremely difficult and likely would have required extensive experimentation.
*   **Lack of Specific Fine-Tuning:** The base `txgemma-2b-predict` model, while pre-trained on a vast corpus, was not specifically fine-tuned for the task of NOAEL prediction from SEND data summaries. Achieving reliable performance on specialized tasks like quantitative prediction often necessitates fine-tuning on a targeted dataset with examples of the desired input-output behavior.

Consequently, the architecture was pivoted to a more conventional and reliable approach using **supervised machine learning**. This allows explicit numerical and categorical features engineered directly from the structured SEND data to be fed into a model (like XGBoost) specifically designed and trained for the regression task of predicting the numerical NOAEL value.

This involved:
    *   Extensive feature engineering to create numerical/categorical vectors from SEND domains (DM, EX, LB, BW, etc.).
    *   Training an XGBoost model on these features (using a placeholder model trained on random data for pipeline testing).
    *   Building API endpoints to serve predictions from this ML model.

## 2. Core Components

The system consists of the following main Python components:

1.  **FastAPI Application:**
    *   Provides the main REST API interface.
    *   Handles incoming HTTP requests.
    *   Defines endpoints for study upload and analysis.
    *   Orchestrates the overall workflow by calling data loading, parsing, and processing modules.
    *   Manages basic configuration (logging, environment variables etc.)

2.  **Data Loading:**
    *   Responsible for locating and reading SEND dataset files (`.xpt`) from a specified study directory.
    *   Parse `.xpt` files into DataFrames.
    *   Handles basic error checking during file loading.

3.  **Domain Parsing:**
    *   Takes raw domain DataFrames as input.
    *   Performs initial cleaning, validation, and extracts key information relevant to the downstream analysis (e.g., standardizing column names, basic type conversions).
    *   Currently focuses on domains needed for the target analysis (DM, EX, TS, BW).

4.  **NOAEL Processor:**
    *   **This is the core logic module.**
    *   Receives parsed domain data.
    *   **Analysis:** Implements the specific analysis strategy (e.g., calculating body weight changes, comparing to controls).
    *   **Prompt Generation:** Constructs a detailed, structured natural language prompt summarizing the analysis findings.
    *   **LLM Interaction:** 
        *   Retrieves OpenRouter API credentials and configuration from environment variables.
        *   Instantiates an OpenAI client endpoint.
        *   Sends the generated prompt to the configured LLM API via the client.
        *   Handles the response from the LLM, including potential errors.
    *   Returns the analysis results, the prompt sent, and the LLM response.

    The screenshot in the previous article shows an example NOAEL prediction for the same Vaccine-1 study.

