import React from 'react';

interface ConfidenceMetricsProps {
  score?: number;
  rationale?: string;
  isLoading?: boolean;
}

const ConfidenceMetrics: React.FC<ConfidenceMetricsProps> = ({
  score = 0,
  rationale = 'N/A',
  isLoading = false,
}) => {
  const scorePercentage = (score * 100).toFixed(0);
  let bgColor = 'bg-red-100';
  let textColor = 'text-red-700';
  if (score > 0.75) {
    bgColor = 'bg-green-100';
    textColor = 'text-green-700';
  } else if (score > 0.5) {
    bgColor = 'bg-yellow-100';
    textColor = 'text-yellow-700';
  }

  return (
    <div className="bg-white shadow-md rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-2">Prediction Confidence</h3>
      {isLoading ? (
        <div className="animate-pulse">
          <div className="h-6 bg-gray-300 rounded w-1/4 mb-2"></div>
          <div className="h-4 bg-gray-300 rounded w-full"></div>
        </div>
      ) : (
        <div>
          <div className={`inline-block px-3 py-1 rounded-full ${bgColor} ${textColor} text-xl font-bold`}>
            {scorePercentage}%
          </div>
          <p className="text-sm text-gray-600 mt-2">Rationale: {rationale}</p>
        </div>
      )}
    </div>
  );
};

export default ConfidenceMetrics; 