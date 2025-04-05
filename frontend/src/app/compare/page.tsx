'use client';

import React from 'react';

const ComparisonPage: React.FC = () => {
  // TODO: Implement study selection and comparison logic
  // - Fetch list of studies
  // - Allow user to select multiple studies
  // - Fetch details/predictions for selected studies
  // - Display comparison view (tables, charts)

  return (
    <div>
      <h2 className="text-2xl font-semibold mb-4">Compare Studies</h2>
      <div className="bg-white shadow-md rounded-lg p-6">
        <p className="text-gray-600">
          Study comparison functionality will be implemented here.
        </p>
        <p className="text-gray-600 mt-2">
          This page will allow selecting multiple uploaded studies and comparing their predicted NOAELs, confidence scores, or other key parameters.
        </p>
        {/* Placeholder for study selection UI */}
        {/* Placeholder for comparison display area */}
      </div>
    </div>
  );
};

export default ComparisonPage; 