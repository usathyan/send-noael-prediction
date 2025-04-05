import React from 'react';

interface StudyDetailsPanelProps {
  studyInfo?: Record<string, any>; // e.g., from TS domain or API response
  isLoading?: boolean;
}

const StudyDetailsPanel: React.FC<StudyDetailsPanelProps> = ({
  studyInfo = {},
  isLoading = false,
}) => {
  // Example placeholder data
  const placeholderInfo = {
    STUDYID: 'Loading...',
    STITLE: 'Loading study title...',
    SPONSOR: 'Loading sponsor...',
    TSPARMCD_Set1: 'Parameter 1',
    TSPARMCD_Set2: 'Parameter 2',
  };

  const displayInfo = isLoading ? placeholderInfo : (Object.keys(studyInfo).length > 0 ? studyInfo : { Message: 'No study details available.' });

  return (
    <div className="bg-white shadow-md rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-2">Study Details</h3>
      {isLoading ? (
        <div className="animate-pulse space-y-2">
          <div className="h-4 bg-gray-300 rounded w-1/4"></div>
          <div className="h-4 bg-gray-300 rounded w-3/4"></div>
          <div className="h-4 bg-gray-300 rounded w-1/2"></div>
          <div className="h-4 bg-gray-300 rounded w-2/3"></div>
        </div>
      ) : (
        <dl className="space-y-1 text-sm">
          {Object.entries(displayInfo).map(([key, value]) => (
            <div key={key} className="flex">
              <dt className="font-medium text-gray-600 w-1/3 truncate">{key}:</dt>
              <dd className="text-gray-800 w-2/3">{String(value)}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
};

export default StudyDetailsPanel; 