import React from 'react';
import { Tool, InsightResult } from '../types';
import { SparklesIcon } from './icons/SparklesIcon';
import { LightbulbIcon } from './icons/LightbulbIcon';

interface ToolInsightsProps {
  selectedTools: Tool[];
  query: string;
  onQueryChange: (query: string) => void;
  onGenerate: () => void;
  result: InsightResult | null;
  isLoading: boolean;
  error: string | null;
}

export const ToolInsights: React.FC<ToolInsightsProps> = ({
  selectedTools,
  query,
  onQueryChange,
  onGenerate,
  result,
  isLoading,
  error,
}) => {
  const isButtonDisabled = isLoading || selectedTools.length === 0;

  return (
    <div className="p-6 bg-slate-800/50 border border-slate-700 rounded-xl">
      <div className="flex items-center gap-3 mb-4">
        <LightbulbIcon className="w-6 h-6 text-yellow-400" />
        <h2 className="text-2xl font-bold text-slate-100">Tool Insights</h2>
      </div>
       <p className="text-sm text-slate-400 mb-6">
        Ask Gemini a question about your selected tools. It will use Google Search to provide up-to-date answers.
      </p>

      <div className="flex flex-col sm:flex-row gap-2 mb-4">
        <input
          type="text"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          placeholder="e.g., Compare LangChain and AutoGen"
          className="flex-grow px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-slate-200 focus:ring-cyan-500 focus:border-cyan-500 transition disabled:opacity-50"
          disabled={selectedTools.length === 0}
        />
        <button
          onClick={onGenerate}
          disabled={isButtonDisabled}
          className="inline-flex items-center justify-center gap-2 px-4 py-2 bg-sky-500/10 text-sky-400 hover:bg-sky-500/20 rounded-md transition-colors text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <SparklesIcon className="w-4 h-4" />
          {isLoading ? 'Getting Insights...' : 'Get Insights'}
        </button>
      </div>
      {selectedTools.length === 0 && <p className="text-sm text-slate-500 mb-4 -mt-2">Select at least one tool to get insights.</p>}

      <div className="mt-4 min-h-[10rem]">
        {isLoading ? (
          <div className="flex items-center justify-center h-full text-slate-400">
            <div className="text-center">
              <svg className="animate-spin h-8 w-8 text-sky-400 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <p className="mt-4">Searching the web and generating insights...</p>
            </div>
          </div>
        ) : error ? (
          <div className="bg-red-900/50 border border-red-700 text-red-300 p-4 rounded-lg">{error}</div>
        ) : result ? (
          <div className="bg-slate-900/50 p-4 rounded-lg">
            <p className="text-slate-300 whitespace-pre-wrap">{result.text}</p>
            {result.sources && result.sources.length > 0 && (
              <div className="mt-6 border-t border-slate-700 pt-4">
                <h4 className="text-sm font-semibold text-slate-400 mb-2">Sources:</h4>
                <ul className="space-y-1">
                  {result.sources.map((source, index) => (
                    <li key={index}>
                      <a
                        href={source.web.uri}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm text-cyan-400 hover:text-cyan-300 hover:underline truncate"
                      >
                        {source.web.title || source.web.uri}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-center text-slate-500 border-2 border-dashed border-slate-700 rounded-lg">
            <p>Insights from Gemini will appear here.</p>
          </div>
        )}
      </div>
    </div>
  );
};
