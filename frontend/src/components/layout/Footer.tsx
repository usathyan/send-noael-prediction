import React from 'react';

interface FooterProps {}

const Footer: React.FC<FooterProps> = ({}) => {
  return (
    <footer className="bg-gray-200 text-gray-600 p-4 text-center text-sm mt-auto">
      © {new Date().getFullYear()} SEND NOAEL Prediction Tool. All rights reserved.
      {/* Add other footer info if needed */}
    </footer>
  );
};

export default Footer; 