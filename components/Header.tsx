import React from 'react';
import { ResetIcon } from './icons/ResetIcon';

interface HeaderProps {
  onReset: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onReset }) => {
  return (
    <header className="bg-slate-900/70 backdrop-blur-lg border-b border-slate-700/50 sticky top-0 z-10">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <div className="text-left">
          <h1 className="text-2xl sm:text-3xl font-bold bg-gradient-to-r from-cyan-400 via-sky-400 to-blue-500 text-transparent bg-clip-text">
            AI Agent Colab Starter
          </h1>
          <p className="hidden sm:block mt-1 text-slate-400 max-w-2xl">
            Select your desired tools to instantly generate a Google Colab setup script.
          </p>
        </div>
        <button
          onClick={onReset}
          className="inline-flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700/50 border border-slate-700 rounded-md transition-colors text-sm font-semibold text-slate-300 hover:text-slate-100"
          title="Reset all selections and inputs"
        >
          <ResetIcon className="w-4 h-4" />
          <span className="hidden sm:inline">Reset</span>
        </button>
      </div>
    </header>
  );
};