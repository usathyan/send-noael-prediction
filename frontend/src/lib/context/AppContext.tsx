'use client'; // Required for Context API in App Router

import React, { createContext, useContext, useReducer, ReactNode, Dispatch } from 'react';
import { PredictionResponse, UploadResponse } from '@/lib/api-client'; // Import types from API client

// --- State Shape ---
interface Study {
  id: string; // Corresponds to study_id from UploadResponse
  uploadDate: Date;
  // Add other relevant study metadata if needed
}

interface AppState {
  isLoading: boolean;
  error: string | null;
  studies: Study[]; // List of uploaded studies
  selectedStudyId: string | null;
  currentPrediction: PredictionResponse | null; // Result for the selected study
}

// --- Actions --- Define actions for the reducer
type Action = 
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'ADD_STUDY'; payload: UploadResponse }
  | { type: 'SET_STUDIES'; payload: Study[] } // For loading initial studies if implemented
  | { type: 'SET_SELECTED_STUDY'; payload: string | null }
  | { type: 'SET_PREDICTION_RESULT'; payload: PredictionResponse | null };

// --- Reducer Function ---
const initialState: AppState = {
  isLoading: false,
  error: null,
  studies: [],
  selectedStudyId: null,
  currentPrediction: null,
};

const appReducer = (state: AppState, action: Action): AppState => {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload, error: null }; // Clear error on new loading
    case 'SET_ERROR':
      return { ...state, isLoading: false, error: action.payload };
    case 'ADD_STUDY':
      const newStudy: Study = {
        id: action.payload.study_id,
        uploadDate: new Date(),
      };
      // Avoid adding duplicates
      if (state.studies.some(s => s.id === newStudy.id)) {
        return state; 
      }
      return { 
        ...state, 
        studies: [...state.studies, newStudy], 
        isLoading: false, 
        error: null 
      };
    case 'SET_STUDIES': // Could be used for loading persisted state
        return { ...state, studies: action.payload };
    case 'SET_SELECTED_STUDY':
      return { ...state, selectedStudyId: action.payload, currentPrediction: null, error: null }; // Clear prediction on selection change
    case 'SET_PREDICTION_RESULT':
      // Ensure the prediction matches the selected study
      if (state.selectedStudyId === action.payload?.study_id) {
          return { ...state, currentPrediction: action.payload, isLoading: false, error: null };
      } 
      // If prediction doesn't match selected study (e.g., race condition), ignore it
      console.warn('Received prediction for non-selected study.');
      return { ...state, isLoading: false }; // Still turn off loading

    default:
      return state;
  }
};

// --- Context Definition ---
interface AppContextProps {
  state: AppState;
  dispatch: Dispatch<Action>;
}

const AppContext = createContext<AppContextProps | undefined>(undefined);

// --- Context Provider Component ---
interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
};

// --- Custom Hook for Consuming Context ---
export const useAppContext = (): AppContextProps => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
}; 