import React from 'react';

interface NOAELSummaryCardProps {
  studyId?: string;
  predictedNoael?: number | string;
  units?: string;
  confidence?: number;
  isLoading?: boolean;
}

const NOAELSummaryCard: React.FC<NOAELSummaryCardProps> = ({
  studyId = 'N/A',
  predictedNoael = '---',
  units = 'mg/kg',
  confidence = 0,
  isLoading = false,
}) => {
  const confidencePercentage = (confidence * 100).toFixed(0);

  return (
    <div className="bg-white shadow-md rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-2">Overall NOAEL Summary</h3>
      {isLoading ? (
        <div className="animate-pulse">
          <div className="h-4 bg-gray-300 rounded w-3/4 mb-2"></div>
          <div className="h-6 bg-gray-300 rounded w-1/2 mb-3"></div>
          <div className="h-4 bg-gray-300 rounded w-1/4"></div>
        </div>
      ) : (
        <div>
          <p className="text-sm text-gray-500">Study: {studyId}</p>
          <p className="text-2xl font-bold my-1">
            {predictedNoael} <span className="text-lg font-normal">{units}</span>
          </p>
          <p className="text-sm text-gray-600">Confidence: {confidencePercentage}%</p>
        </div>
      )}
    </div>
  );
};

export default NOAELSummaryCard; 