import React from 'react';

interface Endpoint {
  id: string;
  name: string;
  domain: string; // e.g., LB, CL, BW
}

interface EndpointSelectionPanelProps {
  endpoints?: Endpoint[];
  selectedEndpoints?: string[]; // List of selected endpoint IDs
  onSelectionChange?: (selectedIds: string[]) => void;
  isLoading?: boolean;
}

const EndpointSelectionPanel: React.FC<EndpointSelectionPanelProps> = ({
  endpoints = [],
  selectedEndpoints = [],
  onSelectionChange = () => {},
  isLoading = false,
}) => {
  // Example placeholder data
  const placeholderEndpoints: Endpoint[] = [
    { id: 'LB_ALT', name: 'Alanine Aminotransferase', domain: 'LB' },
    { id: 'LB_AST', name: 'Aspartate Aminotransferase', domain: 'LB' },
    { id: 'CL_OBS1', name: 'Clinical Observation 1', domain: 'CL' },
    { id: 'BW_BWT', name: 'Body Weight', domain: 'BW' },
  ];

  const displayEndpoints = endpoints.length > 0 ? endpoints : placeholderEndpoints;

  const handleCheckboxChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { value, checked } = event.target;
    const newSelection = checked
      ? [...selectedEndpoints, value]
      : selectedEndpoints.filter(id => id !== value);
    onSelectionChange(newSelection);
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-2">Select Endpoints</h3>
      {isLoading ? (
        <div className="space-y-2">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="animate-pulse flex items-center">
              <div className="h-4 w-4 bg-gray-300 rounded mr-2"></div>
              <div className="h-4 bg-gray-300 rounded w-3/4"></div>
            </div>
          ))}
        </div>
      ) : (
        <ul className="space-y-1 max-h-60 overflow-y-auto">
          {displayEndpoints.map((endpoint) => (
            <li key={endpoint.id} className="flex items-center">
              <input
                type="checkbox"
                id={`endpoint-${endpoint.id}`}
                value={endpoint.id}
                checked={selectedEndpoints.includes(endpoint.id)}
                onChange={handleCheckboxChange}
                className="mr-2 h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <label htmlFor={`endpoint-${endpoint.id}`} className="text-sm">
                {endpoint.name} <span className="text-gray-500">({endpoint.domain})</span>
              </label>
            </li>
          ))}
          {displayEndpoints.length === 0 && <p className="text-gray-500 text-sm">No endpoints available.</p>}
        </ul>
      )}
    </div>
  );
};

export default EndpointSelectionPanel; 