import React from 'react';

interface AnalysisItem {
  id: string;
  studyId: string;
  date: string;
  status: 'Completed' | 'Running' | 'Failed';
}

interface RecentAnalysesProps {
  analyses?: AnalysisItem[];
  isLoading?: boolean;
}

const RecentAnalyses: React.FC<RecentAnalysesProps> = ({
  analyses = [],
  isLoading = false,
}) => {
  // Example placeholder data
  const placeholderAnalyses: AnalysisItem[] = [
    { id: '1', studyId: 'STUDY-ABC', date: '2024-04-05', status: 'Completed' },
    { id: '2', studyId: 'STUDY-XYZ', date: '2024-04-04', status: 'Failed' },
    { id: '3', studyId: 'STUDY-123', date: '2024-04-03', status: 'Completed' },
  ];

  const displayAnalyses = analyses.length > 0 ? analyses : placeholderAnalyses;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Completed': return 'text-green-600';
      case 'Running': return 'text-blue-600';
      case 'Failed': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-2">Recent Analyses</h3>
      {isLoading ? (
        <ul className="space-y-2">
          {[...Array(3)].map((_, i) => (
            <li key={i} className="animate-pulse flex justify-between items-center">
              <div className="h-4 bg-gray-300 rounded w-1/2"></div>
              <div className="h-4 bg-gray-300 rounded w-1/4"></div>
            </li>
          ))}
        </ul>
      ) : (
        <ul className="space-y-1">
          {displayAnalyses.map((analysis) => (
            <li key={analysis.id} className="flex justify-between items-center text-sm p-1 hover:bg-gray-50 rounded">
              <span>
                <a href={`/studies/${analysis.studyId}`} className="text-blue-600 hover:underline">{analysis.studyId}</a>
                <span className="text-gray-500 ml-2">({analysis.date})</span>
              </span>
              <span className={`font-medium ${getStatusColor(analysis.status)}`}>{analysis.status}</span>
            </li>
          ))}
          {displayAnalyses.length === 0 && <p className="text-gray-500 text-sm">No recent analyses.</p>}
        </ul>
      )}
    </div>
  );
};

export default RecentAnalyses; 