# TODO - SEND NOAEL Analysis API (TxGemma Demo)

This file outlines the remaining tasks to complete the focused TxGemma demo API.

## Core Implementation

1.  [ ] **Implement `python/processing/noael_processor.py`:**
    *   [ ] Add function `process_study_for_txgemma(parsed_data: Dict, study_id: str) -> Dict` (consider making it `async`).
    *   [ ] Implement Body Weight (BW) analysis:
        *   Extract relevant columns (USUBJID, BWSTRESN, BWDY) from `parsed_data['bw']`.
        *   Clean data (numeric conversions, dropna).
        *   Identify dose groups by merging with `parsed_data['ex']` (using `_extract_numeric_dose` logic).
        *   Identify control group (dose == 0 or min dose).
        *   Calculate % change from baseline for each subject.
        *   Aggregate mean % change per dose group at key time points (e.g., study end).
        *   (Optional) Perform simple statistical comparison (e.g., t-test vs. control if feasible).
    *   [ ] **Generate LLM Prompt:** Create a structured string (`llm_prompt`) summarizing the study info (DM, EX, TS) and the BW analysis results.
    *   [ ] **Implement LLM Interaction:**
        *   Import `google.generativeai` and `os`.
        *   Retrieve API key (`LLM_API_KEY`) and model name (`LLM_MODEL_NAME`) from environment.
        *   Instantiate `genai.GenerativeModel`.
        *   Call `model.generate_content()` (or `generate_content_async`) with the prompt.
        *   Include `try...except` block for API call errors.
    *   [ ] **Format Return Value:** Return a dictionary containing `study_id`, `status`, `analysis_type`, `llm_prompt`, `llm_response`, `error`.

2.  [ ] **Create Placeholder Module (`python/api/placeholder_processor.py`):**
    *   Create this temporary file with a dummy `async def process_study_for_txgemma(...)` function that returns a basic placeholder dictionary. This allows `main.py` to import something and the server to start *before* the real processor is finished.
    *   Remove this file and the corresponding import in `main.py` once the real `noael_processor.py` is implemented.

3.  [ ] **Finalize API (`python/api/main.py`):**
    *   Replace the placeholder import/call with the actual import/call to `python.processing.noael_processor.process_study_for_txgemma`.
    *   Ensure the endpoint correctly handles the `async` nature if the processor function is async.

## Configuration & Testing

4.  [ ] **Create `.env.example` file:** Add an example file showing `LLM_API_KEY=` and optionally `LLM_MODEL_NAME=`.
5.  [ ] **Test API Endpoints:**
    *   Use `curl` or API client (like Postman or Insomnia) to test `/upload/` with a CBER study zip.
    *   Test `/analyze_noael/{study_id}`.
    *   Verify correct prompt generation and LLM response (or errors) in the JSON output and backend logs.
6.  [ ] **Refine Prompt:** Adjust the prompt structure in `noael_processor.py` based on initial LLM results to improve reasoning and NOAEL assessment accuracy.

## Documentation

7.  [x] Update `README.md` (Done).
8.  [x] Update `Architecture.md` (Done).
9.  [ ] Add docstrings to new functions/modules.

## Optional Enhancements

10. [ ] Implement analysis for other endpoints (LB, MI, CL, OM).
11. [ ] Add more robust statistical analysis.
12. [ ] Implement asynchronous LLM calls for better performance.
13. [ ] Add unit/integration tests. 