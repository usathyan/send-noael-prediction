'use client';

import React, { useState, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation'; // For potential redirection after upload
import { useAppContext } from '@/lib/context/AppContext';
import { uploadStudy } from '@/lib/api-client';

// Define interfaces for the API responses
interface UploadResponseData {
  message: string;
  study_id: string;
  study_path: string;
}

interface NoaelResult {
  predicted_noael: number | null;
  units: string;
  model_used: string;
  status: string;
}

interface PredictionResponse {
  study_id: string;
  noael_result: NoaelResult;
  confidence: number | null;
  error: string | null;
}

const ImportPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadedStudyId, setUploadedStudyId] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);

  const { dispatch } = useAppContext();
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null); // Ref for resetting file input

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setSelectedFile(event.target.files[0]);
      setUploadedStudyId(null); // Reset on new file selection
      setPredictionResult(null);
      setError(null);
      setUploadMessage(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError(null);
    setPredictionResult(null);
    setUploadMessage(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Ensure this URL matches your backend configuration (CORS)
      const response = await fetch('http://127.0.0.1:8000/upload/', {
        method: 'POST',
        body: formData,
      });
      const data: UploadResponseData | { detail: string } = await response.json();

      if (!response.ok) {
        const errorDetail = (data as { detail: string }).detail || `Upload failed: ${response.statusText}`;
        throw new Error(errorDetail);
      }
      
      const successData = data as UploadResponseData;
      setUploadedStudyId(successData.study_id);
      setUploadMessage(successData.message);

    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!uploadedStudyId) return;
    setIsLoading(true);
    setError(null);
    setPredictionResult(null); // Clear previous results

    try {
      const response = await fetch(`http://127.0.0.1:8000/predict/${uploadedStudyId}`, {
        method: 'POST',
      });
      const data: PredictionResponse | { detail: string } = await response.json();

      if (!response.ok) {
         const errorDetail = (data as { detail: string }).detail || `Prediction failed: ${response.statusText}`;
         // Set error state and potentially a failed prediction result
         setError(errorDetail);
         setPredictionResult({
             study_id: uploadedStudyId,
             noael_result: { predicted_noael: null, units: 'N/A', model_used: 'N/A', status: 'Failed'},
             confidence: null,
             error: errorDetail
         });
        return; // Stop processing on error
      }
      
      const successData = data as PredictionResponse;
      setPredictionResult(successData);
       // Set top-level error state if the response itself indicates an error
       if (successData.error) {
           setError(`Prediction completed with error: ${successData.error}`);
       }

    } catch (err: any) {
        setError(err.message);
        // Also update predictionResult state to reflect critical failure
        setPredictionResult({
            study_id: uploadedStudyId,
            noael_result: { predicted_noael: null, units: 'N/A', model_used: 'N/A', status: 'Failed'},
            confidence: null,
            error: err.message
        });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-4">
      <h1 className="text-xl font-semibold mb-4">Import & Predict SEND Study</h1>
      
      <div className="mb-4">
        <label htmlFor="file-upload" className="block mb-2 text-sm font-medium text-gray-900">
          Upload SEND Study Zip File
        </label>
        <input 
          id="file-upload"
          ref={fileInputRef}
          type="file" 
          accept=".zip" 
          onChange={handleFileChange} 
          className="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none p-2"
        />
      </div>

      <button 
        onClick={handleUpload} 
        disabled={!selectedFile || isLoading}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50 disabled:cursor-not-allowed mr-2"
      >
        {isLoading && !predictionResult ? 'Uploading...' : 'Upload Study'}
      </button>

      {uploadedStudyId && (
        <button 
          onClick={handlePredict} 
          disabled={isLoading}
          className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading && predictionResult === null ? 'Predicting...' : `Predict NOAEL for ${uploadedStudyId}`}
        </button>
      )}

      {/* Display Messages/Errors/Results */}
      {isLoading && <p className="mt-4 text-gray-600">Loading...</p>}
      {uploadMessage && <p className="mt-4 text-green-600">{uploadMessage}</p>}
      {error && <p className="mt-4 text-red-600">Error: {error}</p>}
      
      {predictionResult && (
        <div className="mt-6 p-4 border rounded shadow-md bg-white">
          <h3 className="text-lg font-semibold mb-2">Prediction Results for {predictionResult.study_id}</h3>
          <p><span className="font-medium">Status:</span> {predictionResult.noael_result.status}</p>
          {predictionResult.error ? (
              <p className="text-red-600"><span className="font-medium">Details:</span> {predictionResult.error}</p>
          ) : (
            <> 
              <p><span className="font-medium">Predicted NOAEL:</span> {predictionResult.noael_result.predicted_noael?.toFixed(2) ?? 'N/A'}</p>
              <p><span className="font-medium">Units:</span> {predictionResult.noael_result.units}</p>
              <p><span className="font-medium">Model Used:</span> {predictionResult.noael_result.model_used}</p>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default ImportPage; 