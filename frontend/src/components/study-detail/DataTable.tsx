import React from 'react';

interface DataTableProps {
  // Data for the table (e.g., specific domain data or findings)
  data?: Record<string, any>[]; // Array of objects
  columns?: { key: string; header: string }[]; // Column definitions
  isLoading?: boolean;
}

const DataTable: React.FC<DataTableProps> = ({
  data = [],
  columns = [],
  isLoading = false,
}) => {
  // Example placeholder columns if none provided
  const placeholderColumns = data.length > 0 ? Object.keys(data[0]).map(key => ({ key, header: key })) : [];
  const displayColumns = columns.length > 0 ? columns : placeholderColumns;

  return (
    <div className="bg-white shadow-md rounded-lg p-4 overflow-x-auto">
      <h3 className="text-lg font-semibold mb-2">Data Table</h3>
      {isLoading ? (
        <div className="animate-pulse">
          {/* Simulate table loading */}
          <div className="h-8 bg-gray-300 rounded mb-2 w-full"></div>
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-6 bg-gray-200 rounded mb-1 w-full"></div>
          ))}
        </div>
      ) : (
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="bg-gray-50">
            <tr>
              {displayColumns.map((col) => (
                <th
                  key={col.key}
                  scope="col"
                  className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {col.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {data.length > 0 ? (
              data.map((row, rowIndex) => (
                <tr key={rowIndex} className="hover:bg-gray-50">
                  {displayColumns.map((col) => (
                    <td key={col.key} className="px-4 py-2 whitespace-nowrap">
                      {String(row[col.key] ?? '')} {/* Basic rendering */}
                    </td>
                  ))}
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={displayColumns.length} className="px-4 py-4 text-center text-gray-500">
                  No data available.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default DataTable; 