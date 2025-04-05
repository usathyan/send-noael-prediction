import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { AppProvider } from '@/lib/context/AppContext';

// Import layout components
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import MainContent from "@/components/layout/MainContent";
import ContextPanel from "@/components/layout/ContextPanel"; // Optional context panel
import Footer from "@/components/layout/Footer";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SEND NOAEL Prediction",
  description: "Predict NOAEL from SEND datasets using TxGemma",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full bg-gray-100">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased h-full flex flex-col`}
      >
        <AppProvider>
          <Header />
          <div className="flex flex-1 overflow-hidden"> {/* Main content area with flex */}          
            <Sidebar />
            <div className="flex flex-1 overflow-x-hidden overflow-y-auto"> {/* Scrollable main area */}            
              <MainContent>{children}</MainContent>
              {/* ContextPanel could be conditionally rendered based on route/state */}
              {/* <ContextPanel /> */}
            </div>
          </div>
          {/* Optional Footer - Uncomment if needed */}
          {/* <Footer /> */}
        </AppProvider>
      </body>
    </html>
  );
}
