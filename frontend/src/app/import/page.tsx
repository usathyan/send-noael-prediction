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

// Interfaces for the Demo endpoint response
interface DemoNoaelResult {
    overall_noael: number | null;
    dose_units: string;
    analysis_summary?: Record<string, any>; // Optional, from raw_results if needed
    per_endpoint_noael?: Record<string, any>; // Optional, from raw_results if needed
    summary_prompt?: string;
    simulated_response?: string;
}

interface DemoResponse {
    study_id: string;
    demo_name: string;
    results?: DemoNoaelResult; 
    raw_results?: Record<string, any>;
    error?: string | null;
}

const ImportPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadedStudyId, setUploadedStudyId] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  
  // State for TxGemma Demo
  const [demoResult, setDemoResult] = useState<DemoResponse | null>(null);
  const [isDemoLoading, setIsDemoLoading] = useState(false);
  const [demoError, setDemoError] = useState<string | null>(null); // Separate error for demo

  const { dispatch } = useAppContext();
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null); // Ref for resetting file input

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setSelectedFile(event.target.files[0]);
      setUploadedStudyId(null); // Reset on new file selection
      setPredictionResult(null);
      setDemoResult(null);
      setError(null);
      setDemoError(null);
      setUploadMessage(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError(null);
    setDemoError(null);
    setPredictionResult(null);
    setDemoResult(null);
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
    setDemoError(null);
    setPredictionResult(null); // Clear previous results
    setDemoResult(null); // Clear demo results too

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

  const handleRunDemo = async () => {
      if (!uploadedStudyId) return;
      setIsDemoLoading(true);
      setDemoError(null);
      setDemoResult(null); // Clear previous demo results
      
      try {
          const response = await fetch(`http://127.0.0.1:8000/predict/${uploadedStudyId}/txgemma_demos/noael_determination`, {
              method: 'POST',
          });
          const data: DemoResponse | { detail: string } = await response.json();
          
          if (!response.ok) {
              const errorDetail = (data as { detail: string }).detail || `Demo failed: ${response.statusText}`;
              throw new Error(errorDetail);
          }
          
          const successData = data as DemoResponse;
          setDemoResult(successData);
          // Check for errors returned within the demo response payload
          if(successData.error) {
              setDemoError(successData.error);
          }

      } catch (err: any) {
          setDemoError(err.message);
          // Optionally set a basic demoResult structure indicating failure
          setDemoResult({
              study_id: uploadedStudyId,
              demo_name: "Automated NOAEL Determination (Simulated)",
              error: err.message
          });
      } finally {
          setIsDemoLoading(false);
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
        disabled={!selectedFile || isLoading || isDemoLoading}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50 disabled:cursor-not-allowed mr-2"
      >
        {isLoading ? 'Working...' : 'Upload Study'}
      </button>

      {uploadedStudyId && (
        <>
          <button 
            onClick={handlePredict} 
            disabled={isLoading || isDemoLoading}
            className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50 disabled:cursor-not-allowed mr-2"
          >
            {isLoading ? 'Working...' : `Predict NOAEL (ML) for ${uploadedStudyId}`}
          </button>
          
          <button 
            onClick={handleRunDemo} 
            disabled={isLoading || isDemoLoading}
            className="bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isDemoLoading ? 'Running Demo...' : `Run NOAEL Demo (Stats-Based)`}
          </button>
        </>
      )}

      {/* Display Messages/Errors/Results */}
      {(isLoading || isDemoLoading) && <p className="mt-4 text-gray-600">Loading...</p>}
      {uploadMessage && <p className="mt-4 text-green-600">{uploadMessage}</p>}
      {/* Display general or ML prediction errors */}
      {error && <p className="mt-4 text-red-600">ML Prediction Error: {error}</p>}
      {/* Display demo-specific errors */}
      {demoError && <p className="mt-4 text-orange-600">Demo Error: {demoError}</p>}
      
      {predictionResult && (
        <div className="mt-6 p-4 border rounded shadow-md bg-white">
          <h3 className="text-lg font-semibold mb-2">ML Model Prediction for {predictionResult.study_id}</h3>
           {predictionResult.error ? (
              <p className="text-red-600"><span className="font-medium">Status:</span> Failed - {predictionResult.error}</p>
          ) : (
            <> 
              <p><span className="font-medium">Status:</span> {predictionResult.noael_result.status}</p>
              <p><span className="font-medium">Predicted NOAEL:</span> {predictionResult.noael_result.predicted_noael?.toFixed(2) ?? 'N/A'}</p>
              <p><span className="font-medium">Units:</span> {predictionResult.noael_result.units}</p>
              <p><span className="font-medium">Model Used:</span> {predictionResult.noael_result.model_used}</p>
            </>
          )}
        </div>
      )}
      
      {demoResult && (
        <div className="mt-6 p-4 border border-purple-300 rounded shadow-md bg-purple-50">
           <h3 className="text-lg font-semibold mb-2">{demoResult.demo_name} Results for {demoResult.study_id}</h3>
           {demoResult.error ? (
               <p className="text-orange-700"><span className="font-medium">Status:</span> Failed - {demoResult.error}</p>
           ) : demoResult.results ? (
             <> 
               <p><span className="font-medium">Statistically Determined NOAEL:</span> {demoResult.results.overall_noael ?? 'Undetermined'} {demoResult.results.dose_units}</p>
                <p className="mt-2 font-medium">Simulated LLM Response:</p>
                <pre className="mt-1 p-2 bg-gray-100 border rounded text-sm whitespace-pre-wrap break-words">
                    {demoResult.results.simulated_response ?? "No simulated response generated."} 
                </pre>
                {/* Optionally show the summary prompt 
                <p className="mt-2 font-medium">Generated Summary Prompt:</p>
                <pre className="mt-1 p-2 bg-gray-100 border rounded text-sm whitespace-pre-wrap break-words">
                    {demoResult.results.summary_prompt ?? "No prompt generated."}
                </pre>
                */} 
             </>
           ) : (
              <p className="text-gray-600">No results available for the demo.</p> 
           )}
        </div>
      )}
    </div>
  );
};

export default ImportPage; 