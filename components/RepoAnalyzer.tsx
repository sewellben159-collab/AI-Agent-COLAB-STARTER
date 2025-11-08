import React from 'react';
import { AnalysisResult } from '../types';
import { SparklesIcon } from './icons/SparklesIcon';
import { GithubIcon } from './icons/GithubIcon';

interface RepoAnalyzerProps {
  url: string;
  onUrlChange: (url: string) => void;
  onAnalyze: () => void;
  result: AnalysisResult | null;
  isLoading: boolean;
  error: string | null;
}

export const RepoAnalyzer: React.FC<RepoAnalyzerProps> = ({
  url,
  onUrlChange,
  onAnalyze,
  result,
  isLoading,
  error,
}) => {
  return (
    <div className="p-6 bg-slate-800/50 border border-slate-700 rounded-xl">
      <div className="flex items-center gap-3 mb-4">
        <GithubIcon className="w-6 h-6 text-slate-300" />
        <h2 className="text-2xl font-bold text-slate-100">Repository Analyzer</h2>
      </div>
      <p className="text-sm text-slate-400 mb-6">
        Provide a public code repository URL (e.g., from GitHub). Gemini will analyze it and suggest relevant tools for your project.
      </p>

      <div className="flex flex-col sm:flex-row gap-2 mb-4">
        <input
          type="text"
          value={url}
          onChange={(e) => onUrlChange(e.target.value)}
          placeholder="https://github.com/user/repo"
          className="flex-grow px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-slate-200 focus:ring-cyan-500 focus:border-cyan-500 transition"
        />
        <button
          onClick={onAnalyze}
          disabled={isLoading || !url}
          className="inline-flex items-center justify-center gap-2 px-4 py-2 bg-sky-500/10 text-sky-400 hover:bg-sky-500/20 rounded-md transition-colors text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <SparklesIcon className="w-4 h-4" />
          {isLoading ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>

      <div className="mt-4 min-h-[10rem]">
        {isLoading ? (
          <div className="flex items-center justify-center h-full text-slate-400">
            <div className="text-center">
              <svg className="animate-spin h-8 w-8 text-sky-400 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <p className="mt-4">Analyzing repository with Gemini...</p>
            </div>
          </div>
        ) : error ? (
          <div className="bg-red-900/50 border border-red-700 text-red-300 p-4 rounded-lg">{error}</div>
        ) : result ? (
          <div className="bg-slate-900/50 p-4 rounded-lg space-y-4">
              <div>
                <h4 className="text-sm font-semibold text-slate-400 mb-2">Recommended Tools:</h4>
                <p className="text-slate-300">
                    {result.recommendedTools.length > 0 ? result.recommendedTools.join(', ') : 'None from the list were recommended.'}
                </p>
                <p className="text-xs text-slate-500 mt-1">These have been auto-selected for you in the list below.</p>
              </div>
              {result.otherSuggestions && (
                 <div className="border-t border-slate-700 pt-4">
                    <h4 className="text-sm font-semibold text-slate-400 mb-2">Other Suggestions:</h4>
                    <pre className="text-sm text-slate-300 whitespace-pre-wrap font-sans bg-slate-800/50 p-3 rounded-md">{result.otherSuggestions}</pre>
                 </div>
              )}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-center text-slate-500 border-2 border-dashed border-slate-700 rounded-lg">
            <p>Tool suggestions will appear here.</p>
          </div>
        )}
      </div>
    </div>
  );
};
