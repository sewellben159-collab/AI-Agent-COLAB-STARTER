import React, { useMemo } from 'react';
import { Tool, ApiKeyInfo, Notebook } from '../types';
import { ExternalLinkIcon } from './icons/ExternalLinkIcon';
import { InfoIcon } from './icons/InfoIcon';

interface ApiKeyManagerProps {
  selectedTools: Tool[];
  selectedNotebook: Notebook | null;
  apiKeys: Record<string, string>;
  onApiKeyChange: (keyVar: string, value: string) => void;
}

export const ApiKeyManager: React.FC<ApiKeyManagerProps> = ({ selectedTools, selectedNotebook, apiKeys, onApiKeyChange }) => {
  const uniqueApiKeys = useMemo(() => {
    const toolKeys = selectedTools.flatMap(tool => tool.apiKeys || []);
    const notebookKeys = selectedNotebook?.apiKeys || [];
    const allKeys = [...toolKeys, ...notebookKeys];

    const uniqueKeysMap = new Map<string, ApiKeyInfo>();
    allKeys.forEach(keyInfo => {
      if (!uniqueKeysMap.has(keyInfo.envVar)) {
        uniqueKeysMap.set(keyInfo.envVar, keyInfo);
      }
    });
    return Array.from(uniqueKeysMap.values());
  }, [selectedTools, selectedNotebook]);

  if (uniqueApiKeys.length === 0) {
    return null;
  }

  return (
    <div className="p-6 bg-slate-800/50 border border-slate-700 rounded-xl">
      <h2 className="text-2xl font-bold text-slate-100 mb-4">API Key Management</h2>
      <div className="flex items-start gap-3 bg-slate-900/50 p-3 rounded-lg text-sm text-slate-400 mb-6">
        <InfoIcon className="w-5 h-5 mt-0.5 flex-shrink-0 text-sky-400" />
        <p>
          Provide API keys to have them automatically included in the generated code.
          Keys are handled client-side and are not stored.
        </p>
      </div>

      <div className="space-y-4">
        {uniqueApiKeys.map(keyInfo => (
          <div key={keyInfo.envVar}>
            <label htmlFor={keyInfo.envVar} className="block text-sm font-medium text-slate-300 mb-1">
              {keyInfo.label}
            </label>
            <div className="flex items-center gap-2">
              <input
                type="password"
                id={keyInfo.envVar}
                name={keyInfo.envVar}
                value={apiKeys[keyInfo.envVar] || ''}
                onChange={(e) => onApiKeyChange(keyInfo.envVar, e.target.value)}
                placeholder={keyInfo.placeholder}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-slate-200 focus:ring-cyan-500 focus:border-cyan-500 transition"
              />
              <a 
                href={keyInfo.docsUrl} 
                target="_blank" 
                rel="noopener noreferrer" 
                title={`Get your ${keyInfo.label}`}
                className="p-2 bg-slate-700 hover:bg-slate-600 rounded-md transition-colors"
                aria-label={`Get your ${keyInfo.label}`}
              >
                <ExternalLinkIcon className="w-5 h-5 text-slate-400" />
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
