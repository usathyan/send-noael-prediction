import React from 'react';

interface HeaderProps {}

const Header: React.FC<HeaderProps> = ({}) => {
  return (
    <header className="bg-gray-800 text-white p-4 shadow-md sticky top-0 z-50">
      <div className="container mx-auto flex justify-between items-center">
        <h1 className="text-xl font-bold">SEND NOAEL Prediction</h1>
        {/* Placeholder for navigation or user actions */}
        <div>
          {/* e.g., <UserMenu /> */}
        </div>
      </div>
    </header>
  );
};

export default Header; 