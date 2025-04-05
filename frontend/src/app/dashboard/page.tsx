'use client'; // Client component needed for hooks

import React from 'react';
import { useAppContext } from '@/lib/context/AppContext';

// Import Dashboard Components
import NOAELSummaryCard from '@/components/dashboard/NOAELSummaryCard';
import EndpointCoverageVisualization from '@/components/dashboard/EndpointCoverageVisualization';
import DoseResponseCurves from '@/components/dashboard/DoseResponseCurves';
import ConfidenceMetrics from '@/components/dashboard/ConfidenceMetrics';
import RecentAnalyses from '@/components/dashboard/RecentAnalyses';

export default function DashboardPage() {
  return (
    <div>
      <h1>Dashboard</h1>
      <p>Welcome to the SEND NOAEL Prediction dashboard.</p>
      <p>Future content could include summary statistics, recent studies, etc.</p>
    </div>
  );
} 