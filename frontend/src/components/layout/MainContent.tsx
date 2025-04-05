import React from 'react';

interface MainContentProps {
  children: React.ReactNode; // To wrap page content
}

const MainContent: React.FC<MainContentProps> = ({ children }) => {
  return (
    <main className="flex-1 p-6 bg-white">
      {children}
    </main>
  );
};

export default MainContent; 