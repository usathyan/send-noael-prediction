# Frontend Interface Design for NOAEL Visualization

This document outlines the design for a frontend interface that will allow users to visualize NOAEL (No Observed Adverse Effect Level) parameters from SEND (Standard for Exchange of Nonclinical Data) datasets.

## Overview

The frontend interface will provide an intuitive, interactive way to view NOAEL determinations for individual studies or across multiple studies. It will visualize the results from the TxGemma-based prediction model and allow users to explore the underlying data that contributed to the NOAEL determinations.

## User Interface Design

### Application Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                             Header Bar                              │
│                                                                     │
├─────────────────┬───────────────────────────────┬─────────────────┤
│                 │                               │                 │
│                 │                               │                 │
│                 │                               │                 │
│   Navigation    │         Main Content          │    Context      │
│     Sidebar     │             Area              │     Panel       │
│                 │                               │                 │
│                 │                               │                 │
│                 │                               │                 │
├─────────────────┴───────────────────────────────┴─────────────────┤
│                                                                     │
│                             Footer Bar                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Header Bar
- Application title and logo
- User authentication status
- Global actions (settings, help, etc.)
- Study selection dropdown

#### 2. Navigation Sidebar
- Study browser
- View selection (Dashboard, Study Details, Comparison, etc.)
- Data import/export options
- Saved analyses

#### 3. Main Content Area
- Dashboard views
- Visualization panels
- Data tables
- Analysis results

#### 4. Context Panel
- Parameter details
- Confidence metrics
- Model explanations
- Reference information

#### 5. Footer Bar
- Status information
- Version details
- Quick help
- Contact information

## Key Screens and Views

### 1. Dashboard View

The dashboard provides an overview of NOAEL determinations across selected studies or for a single study.

```
┌─────────────────────────────────────────────────────────────────────┐
│ Study: [Study Selection Dropdown ▼]                     User: Admin │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐                   │
│  │                     │  │                     │                   │
│  │   NOAEL Summary     │  │  Endpoint Coverage  │                   │
│  │                     │  │                     │                   │
│  └─────────────────────┘  └─────────────────────┘                   │
│                                                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐                   │
│  │                     │  │                     │                   │
│  │   Dose-Response     │  │  Confidence Metrics │                   │
│  │      Curves         │  │                     │                   │
│  │                     │  │                     │                   │
│  └─────────────────────┘  └─────────────────────┘                   │
│                                                                     │
│  ┌─────────────────────────────────────────────┐                    │
│  │                                             │                    │
│  │            Recent Analyses                  │                    │
│  │                                             │                    │
│  └─────────────────────────────────────────────┘                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Dashboard Components:

1. **NOAEL Summary Card**
   - Overall NOAEL determination
   - Confidence score
   - Key contributing endpoints
   - Quick comparison to historical values

2. **Endpoint Coverage Visualization**
   - Radar chart or heatmap showing endpoint coverage
   - Color-coded by significance to NOAEL determination
   - Interactive elements to drill down into specific endpoints

3. **Dose-Response Curves**
   - Interactive visualization of key dose-response relationships
   - Highlighting of NOAEL threshold
   - Ability to toggle between different endpoints

4. **Confidence Metrics**
   - Visualization of prediction confidence
   - Uncertainty ranges
   - Model agreement indicators
   - Data quality metrics

5. **Recent Analyses**
   - List of recently viewed or analyzed studies
   - Quick access to saved analyses
   - Status indicators for ongoing analyses

### 2. Study Detail View

Provides comprehensive information about NOAEL determination for a single study.

```
┌─────────────────────────────────────────────────────────────────────┐
│ Study: CBER-POC-Pilot-Study1                           User: Admin │
├────────────┬────────────────────────────────────┬──────────────────┤
│            │                                    │                  │
│  Endpoints │   ┌────────────────────────────┐   │  Study Details   │
│            │   │                            │   │                  │
│  □ Body    │   │                            │   │  ID: CBER-001    │
│    Weight  │   │      Dose-Response         │   │  Type: Vaccine   │
│            │   │        Visualization       │   │  Duration: 28d   │
│  □ Clinical│   │                            │   │  Species: Rat    │
│    Signs   │   │                            │   │  Groups: 4       │
│            │   │                            │   │  Animals: 40     │
│  □ Hematol.│   └────────────────────────────┘   │                  │
│            │                                    │  NOAEL           │
│  □ Clinical│   ┌────────────────────────────┐   │  Determination:  │
│    Chem.   │   │                            │   │                  │
│            │   │      Data Table View       │   │  10 mg/kg/day    │
│  □ Histopath│   │                            │   │  (High          │
│            │   │                            │   │   Confidence)    │
│  □ Organ   │   │                            │   │                  │
│    Weights │   │                            │   │  Contributing    │
│            │   │                            │   │  Endpoints:      │
│            │   └────────────────────────────┘   │  - Liver Weight  │
│            │                                    │  - ALT Levels    │
│            │                                    │  - Histopathology│
│            │                                    │                  │
└────────────┴────────────────────────────────────┴──────────────────┘
```

#### Study Detail Components:

1. **Endpoint Selection Panel**
   - Checkboxes for toggling different toxicity endpoints
   - Grouping by domain (clinical chemistry, hematology, etc.)
   - Highlighting of endpoints contributing to NOAEL

2. **Dose-Response Visualization**
   - Interactive plots for selected endpoints
   - Multiple visualization options (line charts, box plots, etc.)
   - Statistical significance indicators
   - NOAEL threshold visualization

3. **Data Table View**
   - Tabular representation of underlying data
   - Sorting and filtering capabilities
   - Statistical summary information
   - Highlighting of adverse effects

4. **Study Details Panel**
   - Study metadata from SEND domains
   - NOAEL determination with confidence level
   - Contributing endpoints with importance scores
   - Links to related studies or historical data

### 3. Multi-Study Comparison View

Allows comparison of NOAEL determinations across multiple studies.

```
┌─────────────────────────────────────────────────────────────────────┐
│ Studies: [Multiple Selection ▼]                        User: Admin │
├────────────┬────────────────────────────────────┬──────────────────┤
│            │                                    │                  │
│  Studies   │   ┌────────────────────────────┐   │  Comparison      │
│            │   │                            │   │  Details         │
│  □ CBER-001│   │                            │   │                  │
│            │   │      NOAEL Comparison      │   │  Studies: 3      │
│  □ CBER-002│   │        Visualization       │   │  Common          │
│            │   │                            │   │  Endpoints: 12   │
│  □ CJ16050 │   │                            │   │                  │
│            │   │                            │   │  Normalization:  │
│  □ Nimble  │   └────────────────────────────┘   │  % of Control    │
│            │                                    │                  │
│  Endpoints │   ┌────────────────────────────┐   │  Statistical     │
│            │   │                            │   │  Method:         │
│  □ Body    │   │                            │   │  ANOVA with      │
│    Weight  │   │     Endpoint Heatmap       │   │  Dunnett's       │
│            │   │                            │   │                  │
│  □ Clinical│   │                            │   │  Notes:          │
│    Signs   │   │                            │   │  Differences in  │
│            │   │                            │   │  study design    │
│  □ Hematol.│   └────────────────────────────┘   │  may affect      │
│            │                                    │  comparability   │
│            │                                    │                  │
└────────────┴────────────────────────────────────┴──────────────────┘
```

#### Multi-Study Comparison Components:

1. **Study Selection Panel**
   - Checkboxes for selecting studies to compare
   - Quick filters (by study type, duration, etc.)
   - Study metadata display

2. **Endpoint Selection Panel**
   - Selection of endpoints common across studies
   - Indication of endpoint availability per study
   - Grouping by domain

3. **NOAEL Comparison Visualization**
   - Bar chart or forest plot of NOAEL values across studies
   - Confidence intervals
   - Normalization options (mg/kg, fold change, etc.)
   - Color coding by confidence level

4. **Endpoint Heatmap**
   - Matrix visualization of endpoint significance across studies
   - Color intensity indicating effect size or statistical significance
   - Clustering of related endpoints
   - Highlighting of consistent findings

5. **Comparison Details Panel**
   - Statistical methods used for comparison
   - Normalization approach
   - Caveats and limitations
   - Export options for comparison results

### 4. Data Import and Configuration View

Interface for importing SEND datasets and configuring the analysis.

```
┌─────────────────────────────────────────────────────────────────────┐
│ Import SEND Dataset                                    User: Admin │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  Drag and drop SEND dataset files here                      │    │
│  │  or click to browse                                         │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌───────────────────────────┐  ┌───────────────────────────────┐  │
│  │                           │  │                               │  │
│  │  Dataset Configuration    │  │  Analysis Options             │  │
│  │                           │  │                               │  │
│  │  Study ID: ____________   │  │  □ Use TxGemma-27B (advanced) │  │
│  │                           │  │  □ Use TxGemma-2B (faster)    │  │
│  │  SEND Version: [3.1 ▼]    │  │                               │  │
│  │                           │  │  □ Compare with historical    │  │
│  │  Species: [Rat ▼]         │  │    control data               │  │
│  │                           │  │                               │  │
│  │  Study Type: [Tox ▼]      │  │  □ Generate detailed report   │  │
│  │                           │  │                               │  │
│  └───────────────────────────┘  └───────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  Validation Results                                         │    │
│  │                                                             │    │
│  │  No issues found. Dataset is ready for analysis.            │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  [Cancel]                                           [Import & Run]  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Import and Configuration Components:

1. **File Upload Area**
   - Drag-and-drop interface for SEND XPT files
   - Directory upload support
   - Progress indicators for large datasets

2. **Dataset Configuration Panel**
   - Study metadata input/confirmation
   - SEND version selection
   - Species and study type specification
   - Additional metadata fields

3. **Analysis Options Panel**
   - Model selection (TxGemma variant)
   - Analysis depth configuration
   - Comparison options
   - Report generation settings

4. **Validation Results Area**
   - Real-time validation of uploaded datasets
   - Error and warning messages
   - Suggestions for resolving issues
   - Dataset completeness indicators

## Interactive Features

### 1. Dose-Response Visualization

Interactive visualization of dose-response relationships with the following features:

- **Zoom and Pan**: Ability to focus on specific dose ranges
- **Endpoint Selection**: Toggle between different endpoints
- **Threshold Adjustment**: Interactive NOAEL threshold line that users can adjust
- **Confidence Intervals**: Toggleable display of confidence intervals
- **Statistical Annotations**: Markers for statistical significance
- **Time Series View**: For endpoints measured at multiple timepoints
- **Group Highlighting**: Ability to highlight specific dose groups
- **Export Options**: Save visualizations as images or data

### 2. Data Filtering and Exploration

Tools for exploring the underlying SEND data:

- **Advanced Filters**: Filter by sex, age, timepoint, etc.
- **Custom Grouping**: Create custom groups for analysis
- **Outlier Identification**: Highlight statistical outliers
- **Data Drill-Down**: Click on summary visualizations to see individual animal data
- **Custom Calculations**: Define custom calculations or transformations
- **Annotation Tools**: Add notes or flags to specific data points
- **Search Functionality**: Search across all data fields

### 3. Comparative Analysis

Features for comparing across studies or endpoints:

- **Side-by-Side Views**: Compare multiple endpoints or studies
- **Difference Plots**: Visualize differences between studies
- **Normalization Options**: Multiple ways to normalize data for comparison
- **Historical Control Ranges**: Overlay historical control ranges
- **Trend Analysis**: Identify trends across related studies
- **Consistency Scoring**: Metrics for consistency of findings

## Technical Implementation

### Frontend Technology Stack

1. **Framework**: Next.js
   - Server-side rendering for improved performance
   - API routes for backend communication
   - Static site generation for documentation pages

2. **UI Components**: React with Tailwind CSS
   - Responsive design for desktop and tablet use
   - Accessibility compliance
   - Theme customization

3. **Visualization Libraries**:
   - Recharts for standard charts
   - D3.js for custom visualizations
   - react-table for data tables

4. **State Management**:
   - React Context API for global state
   - SWR for data fetching and caching

### Backend Integration

The frontend will communicate with the backend prediction model through:

1. **RESTful API**:
   - Endpoints for data upload and validation
   - Analysis request and status checking
   - Result retrieval and formatting

2. **WebSocket Connection**:
   - Real-time updates for long-running analyses
   - Progress indicators
   - Immediate notification of results

### Data Flow

1. User uploads SEND datasets through the interface
2. Frontend validates basic structure and sends to backend
3. Backend processes data and runs TxGemma prediction
4. Results are returned to frontend and cached
5. Frontend renders visualizations and interactive elements
6. User explores results and can export findings

## Responsive Design

The interface will be optimized for:

1. **Desktop**: Full-featured interface with side-by-side panels
2. **Tablet**: Adapted layout with collapsible panels
3. **Large Displays**: Enhanced visualization area for presentation mode

## Accessibility Features

1. **Keyboard Navigation**: Full keyboard accessibility
2. **Screen Reader Support**: ARIA labels and semantic HTML
3. **Color Considerations**: Colorblind-friendly palettes with patterns
4. **Text Scaling**: Support for browser text scaling
5. **Focus Indicators**: Clear visual indicators for keyboard focus

## Export and Sharing Capabilities

1. **Report Generation**:
   - PDF reports with key findings
   - Customizable report templates
   - Inclusion of visualizations and data tables

2. **Data Export**:
   - CSV/Excel export of processed data
   - JSON export of analysis results
   - Image export of visualizations

3. **Sharing Options**:
   - Shareable links to specific analyses
   - Collaboration features for team review
   - Presentation mode for meetings

## Implementation Approach for Cursor Tool

The frontend interface will be implemented using Cursor with the following approach:

1. **Project Setup**:
   - Use Cursor to create a new Next.js project with Tailwind CSS
   - Set up the project structure following Next.js best practices
   - Configure TypeScript for type safety

2. **Component Development**:
   - Leverage Cursor's AI capabilities to generate component templates
   - Implement visualization components with guidance from Cursor
   - Use Cursor for efficient styling with Tailwind CSS

3. **Data Integration**:
   - Implement data processing utilities with Cursor's assistance
   - Create API integration code for backend communication
   - Develop data transformation functions for visualization

4. **Testing and Refinement**:
   - Use Cursor to generate test cases for components
   - Leverage Cursor for debugging and performance optimization
   - Implement responsive design adjustments

## Conclusion

The proposed frontend interface design provides a comprehensive solution for visualizing NOAEL determinations from SEND datasets. It offers intuitive navigation, interactive visualizations, and powerful analysis tools while maintaining accessibility and responsiveness. The implementation approach leverages modern web technologies and the Cursor tool to create an efficient development workflow.
