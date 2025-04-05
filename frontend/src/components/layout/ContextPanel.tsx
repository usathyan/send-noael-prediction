import React from 'react';

interface ContextPanelProps {
  // Props for context-specific info
}

const ContextPanel: React.FC<ContextPanelProps> = ({}) => {
  return (
    <aside className="bg-gray-50 w-72 p-4 h-screen sticky top-0 overflow-y-auto border-l">
      <h2 className="text-lg font-semibold mb-4">Context</h2>
      {/* Placeholder for context-specific details, actions, or visualizations */}
      <p className="text-gray-600">Contextual information will appear here based on selection.</p>
    </aside>
  );
};

export default ContextPanel; 