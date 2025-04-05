# Implementation Guide for NOAEL Prediction Tool in Cursor

This guide provides detailed instructions for implementing the NOAEL (No Observed Adverse Effect Level) prediction tool for SEND (Standard for Exchange of Nonclinical Data) datasets using the Cursor development environment.

## Table of Contents

1. [Introduction](#introduction)
2. [Setting Up the Development Environment](#setting-up-the-development-environment)
3. [Backend Implementation](#backend-implementation)
4. [Frontend Implementation](#frontend-implementation)
5. [Integration and Testing](#integration-and-testing)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)

## Introduction

The NOAEL prediction tool combines TxGemma's machine learning capabilities with a specialized frontend for visualizing toxicology data from SEND datasets. This implementation guide will walk you through the process of building this tool using Cursor, an AI-powered code editor that enhances development productivity.

### Project Overview

The complete system consists of:
- A data processing pipeline for SEND datasets
- A prediction model based on TxGemma
- A Next.js frontend for visualization and analysis
- API endpoints for communication between components

### Prerequisites

- Cursor IDE installed (latest version)
- Node.js (v18+)
- Python (v3.10+)
- Git
- Basic knowledge of React, Next.js, and Python
- Access to TxGemma models via Hugging Face

## Setting Up the Development Environment

### Step 1: Initialize the Project in Cursor

1. Open Cursor and create a new project:
   ```
   /new-project noael-prediction-tool
   ```

2. Initialize a Next.js project with the following command:
   ```
   /cursor-command "Create a new Next.js project with TypeScript, Tailwind CSS, and API routes. Include ESLint configuration."
   ```

3. Set up the project structure:
   ```
   /cursor-command "Create the following directory structure:
   - src/
     - app/
     - components/
     - lib/
     - api/
     - hooks/
     - types/
   - public/
   - python/
     - data_processing/
     - model/
     - api/
   "
   ```

### Step 2: Set Up Python Environment

1. Create a Python virtual environment:
   ```
   /cursor-command "Create a Python virtual environment setup with requirements.txt including pandas, numpy, xport, transformers, fastapi, and uvicorn."
   ```

2. Initialize the Python backend structure:
   ```
   /cursor-command "Create Python files for SEND data processing, TxGemma model integration, and API endpoints."
   ```

## Backend Implementation

### Step 1: SEND Data Processing Pipeline

1. Implement the SEND dataset loader:
   ```
   /cursor-command "Create a Python module for loading and validating SEND datasets in XPT format. Include functions for parsing different SEND domains (DM, EX, LB, CL, etc.)."
   ```

   This will generate a file like `python/data_processing/send_loader.py` with functions for loading and validating SEND datasets.

2. Implement the domain parser:
   ```
   /cursor-command "Create a Python module for extracting relevant information from each SEND domain. Include functions for processing demographics, dosing information, toxicity endpoints, and trial design information."
   ```

   This will generate a file like `python/data_processing/domain_parser.py` with functions for extracting information from SEND domains.

3. Implement the feature extractor:
   ```
   /cursor-command "Create a Python module for transforming raw SEND data into features suitable for TxGemma. Include functions for generating dose-response relationships, calculating statistical measures, normalizing values, and handling missing data."
   ```

   This will generate a file like `python/data_processing/feature_extractor.py` with functions for feature extraction.

### Step 2: TxGemma Model Integration

1. Implement the TxGemma model wrapper:
   ```
   /cursor-command "Create a Python module for integrating with TxGemma models. Include functions for loading models from Hugging Face, formatting inputs, generating predictions, and processing outputs."
   ```

   This will generate a file like `python/model/txgemma_wrapper.py` with functions for working with TxGemma models.

2. Implement the NOAEL calculator:
   ```
   /cursor-command "Create a Python module for determining NOAEL values from model predictions. Include functions for applying statistical methods, calculating NOAEL for each endpoint, determining overall study NOAEL, and generating uncertainty bounds."
   ```

   This will generate a file like `python/model/noael_calculator.py` with functions for calculating NOAEL values.

3. Implement the confidence score generator:
   ```
   /cursor-command "Create a Python module for assessing the reliability of NOAEL predictions. Include functions for calculating confidence scores, identifying confounding factors, and generating explanations."
   ```

   This will generate a file like `python/model/confidence_scorer.py` with functions for generating confidence scores.

### Step 3: API Endpoints

1. Implement the FastAPI application:
   ```
   /cursor-command "Create a FastAPI application with endpoints for uploading SEND datasets, running predictions, and retrieving results."
   ```

   This will generate a file like `python/api/main.py` with a FastAPI application and endpoints.

2. Implement the data validation endpoint:
   ```
   /cursor-command "Create a FastAPI endpoint for validating uploaded SEND datasets. Include functions for checking file format, domain completeness, and data integrity."
   ```

   This will add validation endpoints to the FastAPI application.

3. Implement the prediction endpoint:
   ```
   /cursor-command "Create a FastAPI endpoint for running NOAEL predictions. Include functions for processing uploaded data, running the model, and returning results."
   ```

   This will add prediction endpoints to the FastAPI application.

## Frontend Implementation

### Step 1: Core Components

1. Implement the layout components:
   ```
   /cursor-command "Create React components for the application layout, including Header, Navigation Sidebar, Main Content Area, Context Panel, and Footer."
   ```

   This will generate files in `src/components/layout/` for each layout component.

2. Implement the dashboard components:
   ```
   /cursor-command "Create React components for the dashboard view, including NOAEL Summary Card, Endpoint Coverage Visualization, Dose-Response Curves, Confidence Metrics, and Recent Analyses."
   ```

   This will generate files in `src/components/dashboard/` for each dashboard component.

3. Implement the study detail components:
   ```
   /cursor-command "Create React components for the study detail view, including Endpoint Selection Panel, Dose-Response Visualization, Data Table View, and Study Details Panel."
   ```

   This will generate files in `src/components/study-detail/` for each study detail component.

### Step 2: Visualization Components

1. Implement the dose-response visualization:
   ```
   /cursor-command "Create a React component for interactive dose-response visualization using Recharts. Include features for zooming, panning, endpoint selection, threshold adjustment, and confidence intervals."
   ```

   This will generate a file like `src/components/visualizations/DoseResponseChart.tsx` with an interactive chart component.

2. Implement the endpoint heatmap:
   ```
   /cursor-command "Create a React component for visualizing endpoint significance across studies using a heatmap. Include features for color intensity mapping, clustering, and highlighting."
   ```

   This will generate a file like `src/components/visualizations/EndpointHeatmap.tsx` with a heatmap component.

3. Implement the data table:
   ```
   /cursor-command "Create a React component for displaying tabular data with sorting, filtering, and statistical summary information."
   ```

   This will generate a file like `src/components/visualizations/DataTable.tsx` with a data table component.

### Step 3: Data Management

1. Implement the API client:
   ```
   /cursor-command "Create a TypeScript module for communicating with the backend API. Include functions for uploading datasets, validating data, running predictions, and retrieving results."
   ```

   This will generate a file like `src/lib/api-client.ts` with functions for API communication.

2. Implement the state management:
   ```
   /cursor-command "Create React context providers for managing application state, including study selection, endpoint selection, and analysis results."
   ```

   This will generate files in `src/lib/context/` for state management.

3. Implement the data transformation utilities:
   ```
   /cursor-command "Create TypeScript utilities for transforming API responses into formats suitable for visualization components."
   ```

   This will generate a file like `src/lib/transformers.ts` with data transformation functions.

### Step 4: Pages and Routing

1. Implement the dashboard page:
   ```
   /cursor-command "Create a Next.js page for the dashboard view that integrates the dashboard components and handles data fetching."
   ```

   This will generate a file like `src/app/dashboard/page.tsx` for the dashboard page.

2. Implement the study detail page:
   ```
   /cursor-command "Create a Next.js page for the study detail view that integrates the study detail components and handles data fetching based on study ID."
   ```

   This will generate a file like `src/app/studies/[id]/page.tsx` for the study detail page.

3. Implement the comparison page:
   ```
   /cursor-command "Create a Next.js page for the multi-study comparison view that allows selection of multiple studies and endpoints for comparison."
   ```

   This will generate a file like `src/app/compare/page.tsx` for the comparison page.

4. Implement the import page:
   ```
   /cursor-command "Create a Next.js page for the data import and configuration view with file upload, validation, and analysis options."
   ```

   This will generate a file like `src/app/import/page.tsx` for the import page.

## Integration and Testing

### Step 1: API Integration

1. Connect frontend to backend:
   ```
   /cursor-command "Update the API client to connect to the FastAPI backend. Include error handling, loading states, and response caching."
   ```

   This will update the API client with proper integration code.

2. Implement WebSocket connection:
   ```
   /cursor-command "Create a WebSocket client for real-time updates during long-running analyses. Include connection management, message handling, and reconnection logic."
   ```

   This will generate a file like `src/lib/websocket-client.ts` for WebSocket communication.

### Step 2: Testing

1. Implement unit tests:
   ```
   /cursor-command "Create Jest unit tests for key components and utilities. Include tests for data transformation, visualization rendering, and state management."
   ```

   This will generate test files in `__tests__/` directories throughout the project.

2. Implement integration tests:
   ```
   /cursor-command "Create Cypress integration tests for key user flows, including data import, analysis, and visualization interaction."
   ```

   This will set up Cypress and generate integration test files.

3. Implement API tests:
   ```
   /cursor-command "Create Python tests for the FastAPI endpoints using pytest. Include tests for data validation, prediction, and error handling."
   ```

   This will generate test files in `python/tests/` for API testing.

## Deployment

### Step 1: Backend Deployment

1. Create a Docker configuration:
   ```
   /cursor-command "Create a Dockerfile for the Python backend that includes all dependencies and configuration for production deployment."
   ```

   This will generate a `Dockerfile` for the backend.

2. Create deployment scripts:
   ```
   /cursor-command "Create deployment scripts for the backend, including environment setup, database initialization, and service configuration."
   ```

   This will generate deployment scripts in a `deploy/` directory.

### Step 2: Frontend Deployment

1. Configure Next.js for production:
   ```
   /cursor-command "Update Next.js configuration for production deployment, including environment variables, build optimization, and static asset handling."
   ```

   This will update the Next.js configuration files.

2. Create a deployment workflow:
   ```
   /cursor-command "Create a GitHub Actions workflow for building and deploying the frontend to a hosting service like Vercel or Netlify."
   ```

   This will generate a workflow file in `.github/workflows/`.

## Troubleshooting

### Common Issues and Solutions

1. SEND Dataset Loading Issues:
   - Ensure XPT files are in the correct format
   - Check for missing required domains
   - Verify file permissions

2. TxGemma Integration Issues:
   - Ensure proper API keys and access
   - Check for model version compatibility
   - Verify input formatting

3. Frontend Rendering Issues:
   - Check browser console for errors
   - Verify data structure matches component expectations
   - Test with smaller datasets first

### Performance Optimization

1. Backend Optimization:
   - Implement caching for processed datasets
   - Use batch processing for large files
   - Optimize database queries

2. Frontend Optimization:
   - Implement virtualization for large data tables
   - Use memoization for expensive calculations
   - Implement code splitting for faster initial load

## Using Cursor AI Effectively

### Generating Component Code

When implementing components, use specific prompts to generate code:

```
/cursor-command "Create a React component for a dose-response chart with the following features:
- X-axis for dose levels
- Y-axis for response values
- Multiple series for different endpoints
- Interactive tooltips showing data points
- Threshold line for NOAEL
- Confidence interval shading
- Zoom and pan controls
Use Recharts as the visualization library and TypeScript for type safety."
```

### Debugging with Cursor

When encountering issues, use Cursor to help debug:

```
/cursor-command "Debug the following error in the DoseResponseChart component: 'TypeError: Cannot read property 'data' of undefined'. Suggest potential fixes and explain the root cause."
```

### Refactoring Code

Use Cursor to refactor and improve code quality:

```
/cursor-command "Refactor the following API client code to use React Query for better caching, loading states, and error handling."
```

## Conclusion

By following this implementation guide, you can build a comprehensive NOAEL prediction tool using Cursor's AI-powered development capabilities. The resulting application will provide powerful visualization and analysis tools for toxicology data from SEND datasets, leveraging TxGemma's machine learning capabilities for accurate NOAEL predictions.

Remember to iterate on the implementation based on user feedback and testing results. The modular architecture allows for incremental improvements and extensions as requirements evolve.
