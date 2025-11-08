import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Tool, Notebook } from '../types';
import { generateStarterCode } from '../services/geminiService';
import { ClipboardIcon } from './icons/ClipboardIcon';
import { CheckIcon } from './icons/CheckIcon';
import { ExternalLinkIcon } from './icons/ExternalLinkIcon';
import { SparklesIcon } from './icons/SparklesIcon';

interface OutputPaneProps {
  selectedTools: Tool[];
  selectedNotebook: Notebook | null;
  apiKeys: Record<string, string>;
}

enum Tab {
  Install,
  Gemini,
}

const CodeBlock: React.FC<{ code: string }> = ({ code }) => {
  const [hasCopied, setHasCopied] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code);
    setHasCopied(true);
    setTimeout(() => setHasCopied(false), 2000);
  };

  return (
    <div className="relative bg-slate-900 rounded-lg p-4 h-full">
      <button
        onClick={copyToClipboard}
        className="absolute top-3 right-3 p-2 bg-slate-700 hover:bg-slate-600 rounded-md transition-colors"
        aria-label="Copy to clipboard"
      >
        {hasCopied ? <CheckIcon className="w-5 h-5 text-green-400" /> : <ClipboardIcon className="w-5 h-5 text-slate-400" />}
      </button>
      <pre className="text-sm text-slate-300 overflow-auto h-full pr-12">
        <code>{code}</code>
      </pre>
    </div>
  );
};

export const OutputPane: React.FC<OutputPaneProps> = ({ selectedTools, selectedNotebook, apiKeys }) => {
  const [activeTab, setActiveTab] = useState<Tab>(Tab.Install);
  const [installScript, setInstallScript] = useState('');
  const [geminiCode, setGeminiCode] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const notebookContentWithKeys = useMemo(() => {
    if (!selectedNotebook) return '';
    
    let content = selectedNotebook.content;
    
    if (selectedNotebook.apiKeys) {
      for (const keyInfo of selectedNotebook.apiKeys) {
        const userKey = apiKeys[keyInfo.envVar];
        const placeholder = new RegExp(`"${keyInfo.placeholder}"`, 'g');
        content = content.replace(placeholder, `"${userKey || keyInfo.placeholder}"`);
      }
    }
    // A common placeholder
    content = content.replace(/"your_hf_token_here"/g, `"${apiKeys['HF_TOKEN'] || 'your_hf_token_here'}"`);


    return content;
  }, [selectedNotebook, apiKeys]);


  useEffect(() => {
    if (selectedNotebook) {
        // In notebook mode, default to showing that.
        setActiveTab(Tab.Install); // Or a new "Notebook" tab if we wanted one.
    } else {
        const commands = selectedTools.map(tool => `# ${tool.name}\n${tool.installCommand}`).join('\n\n');
        setInstallScript(commands || '# Select tools to generate the installation script.');
    }
  }, [selectedTools, selectedNotebook]);

  const handleGenerateCode = useCallback(async () => {
    setIsGenerating(true);
    setError(null);
    setGeminiCode('');
    try {
      const code = await generateStarterCode(selectedTools, apiKeys);
      setGeminiCode(code);
    } catch (e: any) {
      setError(e.message || 'An unknown error occurred.');
    } finally {
      setIsGenerating(false);
    }
  }, [selectedTools, apiKeys]);
  
  const createColabUrl = () => {
    const baseUrl = 'https://colab.research.google.com/notebooks/empty.ipynb#create=true&code=';
    
    if (selectedNotebook) {
        return baseUrl + encodeURIComponent(notebookContentWithKeys);
    }

    // Using #@title creates collapsible, titled sections in Colab, simulating separate cells for a better UX.
    const installHeader = `#@title 1. Install Dependencies`;
    const installCell = `${installHeader}\n\n${installScript}`;
    
    let finalCode = installCell;
    
    if (geminiCode) {
      const geminiHeader = `#@title 2. Run Starter Code`;
      const geminiCell = `${geminiHeader}\n\n${geminiCode}`;
      // A visual separator for the raw code view.
      const separator = `\n\n# ==============================================================================`;
      finalCode += separator + `\n\n${geminiCell}`;
    }
    
    return baseUrl + encodeURIComponent(finalCode);
  };
  
  const colabUrl = createColabUrl();


  return (
    <div className="p-1 bg-slate-800/50 border border-slate-700 rounded-xl h-[80vh] flex flex-col sticky top-24">
      <div className="flex justify-between items-center p-4">
        <div className="flex border-b border-slate-700">
          {selectedNotebook ? (
             <TabButton name="Notebook Preview" isActive={true} onClick={() => {}} />
          ) : (
            <>
              <TabButton name="Installation Script" isActive={activeTab === Tab.Install} onClick={() => setActiveTab(Tab.Install)} />
              <TabButton name="Gemini Starter Code" isActive={activeTab === Tab.Gemini} onClick={() => setActiveTab(Tab.Gemini)} />
            </>
          )}
        </div>
        <a 
          href={colabUrl} 
          target="_blank" 
          rel="noopener noreferrer" 
          className="inline-flex items-center gap-2 px-4 py-2 bg-yellow-500/10 text-yellow-400 hover:bg-yellow-500/20 rounded-md transition-colors text-sm font-semibold"
          title="Opens a new Colab notebook with the installation script and any generated Gemini code"
        >
          Open in Colab
          <ExternalLinkIcon className="w-4 h-4" />
        </a>
      </div>
      <div className="flex-grow p-4 min-h-0">
        {selectedNotebook ? (
            <CodeBlock code={notebookContentWithKeys} />
        ) : activeTab === Tab.Install ? (
          <CodeBlock code={installScript} />
        ) : (
          <div className="flex flex-col h-full">
            <div className="flex-shrink-0 mb-4 flex items-center gap-4">
              <button
                onClick={handleGenerateCode}
                disabled={isGenerating || selectedTools.length === 0}
                className="inline-flex items-center gap-2 px-4 py-2 bg-sky-500/10 text-sky-400 hover:bg-sky-500/20 rounded-md transition-colors text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <SparklesIcon className="w-4 h-4" />
                {isGenerating ? 'Generating...' : 'Generate with Gemini'}
              </button>
              {selectedTools.length === 0 && <p className="text-sm text-slate-500">Select at least one tool to generate code.</p>}
            </div>
            <div className="flex-grow min-h-0">
              {isGenerating ? (
                 <div className="flex items-center justify-center h-full text-slate-400">
                    <div className="text-center">
                        <svg className="animate-spin h-8 w-8 text-sky-400 mx-auto" xmlns="http://www.w.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <p className="mt-4">Generating code with Gemini...</p>
                    </div>
                </div>
              ) : error ? (
                <div className="bg-red-900/50 border border-red-700 text-red-300 p-4 rounded-lg">{error}</div>
              ) : geminiCode ? (
                <CodeBlock code={geminiCode} />
              ) : (
                <div className="flex items-center justify-center h-full text-center text-slate-500 border-2 border-dashed border-slate-700 rounded-lg">
                  <p>Click "Generate with Gemini" to create a starter script for your selected tools.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};


const TabButton: React.FC<{name: string, isActive: boolean, onClick: () => void}> = ({ name, isActive, onClick }) => {
    return (
        <button
            onClick={onClick}
            className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 whitespace-nowrap
            ${isActive 
                ? 'border-cyan-400 text-cyan-300' 
                : 'border-transparent text-slate-400 hover:text-slate-200'}`
            }
            disabled={isActive}
        >
            {name}
        </button>
    )
}
