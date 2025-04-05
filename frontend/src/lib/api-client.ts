// Define the base URL for the backend API
// This assumes the Next.js app is served on a different port (e.g., 3000)
// than the FastAPI backend (e.g., 8000) during development.
// In production, this might be the same origin or configured via env variables.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

// --- Types matching Python Pydantic models (can be shared/generated ideally) ---
export interface UploadResponse {
  message: string;
  study_id: string;
  study_path: string;
}

export interface NoaelResult {
  value: number | string | null;
  units: string;
  determination_method: string;
  // Add other relevant fields from Python `calculate_noael_from_prediction`
}

export interface ConfidenceInfo {
  score: number;
  rationale: string;
  factors_considered: string[];
  // Add other relevant fields from Python `generate_confidence_score`
}

export interface PredictionDetails {
  predicted_noael?: number | string;
  units?: string;
  raw_output?: string;
  // Add other intermediate results if needed
}

export interface PredictionResponse {
  study_id: string;
  noael_result: NoaelResult | null;
  confidence: ConfidenceInfo | null;
  prediction_details: PredictionDetails | null;
  error?: string | null;
}

// --- API Client Functions ---

/**
 * Uploads a SEND study zip file to the backend.
 * @param file - The zip file to upload.
 * @returns A promise that resolves to the UploadResponse.
 */
export const uploadStudy = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/upload/`, {
    method: 'POST',
    body: formData,
    // Headers are automatically set for FormData by the browser
  });

  const data = await response.json();

  if (!response.ok) {
    // Throw an error with the detail message from the backend if available
    const errorDetail = data.detail || `HTTP error! status: ${response.status}`;
    throw new Error(errorDetail);
  }

  return data as UploadResponse;
};

/**
 * Triggers the NOAEL prediction pipeline for a given study ID.
 * @param studyId - The ID of the study to run prediction for.
 * @returns A promise that resolves to the PredictionResponse.
 */
export const runPrediction = async (studyId: string): Promise<PredictionResponse> => {
  if (!studyId) {
    throw new Error('Study ID cannot be empty.');
  }

  const response = await fetch(`${API_BASE_URL}/predict/${encodeURIComponent(studyId)}`, {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      // Add any other necessary headers like Authorization if needed later
    },
  });

  const data = await response.json();

  if (!response.ok) {
    // Throw an error with the detail message from the backend if available
    const errorDetail = data.detail || `HTTP error! status: ${response.status}`;
    throw new Error(errorDetail);
  }

  return data as PredictionResponse;
};

// Example of a function to get study details (if needed - requires backend endpoint)
// export const getStudyDetails = async (studyId: string): Promise<any> => {
//   const response = await fetch(`${API_BASE_URL}/studies/${encodeURIComponent(studyId)}`);
//   if (!response.ok) {
//     throw new Error(`HTTP error! status: ${response.status}`);
//   }
//   return response.json();
// }; 