import React from 'react';
import Link from 'next/link';

interface SidebarProps {
  // Props for navigation items, etc.
}

const Sidebar: React.FC<SidebarProps> = ({}) => {
  return (
    <aside className="bg-gray-100 w-64 p-4 h-screen sticky top-0 overflow-y-auto">
      <nav>
        <h2 className="text-lg font-semibold mb-4">Navigation</h2>
        <ul>
          <li className="mb-2">
            <Link href="/dashboard" className="text-blue-600 hover:underline">
              Dashboard
            </Link>
          </li>
          <li className="mb-2">
            <Link href="/import" className="text-blue-600 hover:underline">
              Import Data
            </Link>
          </li>
          <li className="mb-2">
            <Link href="/studies" className="text-blue-600 hover:underline">
              Studies
            </Link>
          </li>
          <li className="mb-2">
            <Link href="/" className="text-blue-600 hover:underline">
              Home
            </Link>
          </li>
          {/* Add more navigation links here using Link */}
        </ul>
      </nav>
    </aside>
  );
};

export default Sidebar; 