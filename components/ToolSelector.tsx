
import React, { useState } from 'react';
import { ToolCategory, Tool } from '../types';

interface AccordionItemProps {
  category: ToolCategory;
  selectedToolIds: Set<string>;
  onToolToggle: (toolId: string) => void;
}

const AccordionItem: React.FC<AccordionItemProps> = ({ category, selectedToolIds, onToolToggle }) => {
  const [isOpen, setIsOpen] = useState(true);

  const categoryToolIds = category.tools.map(t => t.id);
  const selectedInCategoryCount = categoryToolIds.filter(id => selectedToolIds.has(id)).length;

  return (
    <div className="border border-slate-700 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex justify-between items-center p-4 bg-slate-800 hover:bg-slate-700/50 transition-colors"
      >
        <div className="text-left">
            <h3 className="font-semibold text-lg text-slate-100">{category.name}</h3>
            <p className="text-sm text-slate-400">{selectedInCategoryCount} / {category.tools.length} selected</p>
        </div>
        <svg
          className={`w-6 h-6 text-slate-400 transform transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
        </svg>
      </button>
      {isOpen && (
        <div className="bg-slate-800/50 p-4 border-t border-slate-700">
          <div className="space-y-4">
            {category.tools.map(tool => (
              <label key={tool.id} className="flex items-start p-3 rounded-md hover:bg-slate-700/50 cursor-pointer transition-colors">
                <input
                  type="checkbox"
                  className="mt-1 h-4 w-4 rounded border-slate-600 bg-slate-700 text-cyan-500 focus:ring-cyan-500 focus:ring-offset-slate-800"
                  checked={selectedToolIds.has(tool.id)}
                  onChange={() => onToolToggle(tool.id)}
                />
                <div className="ml-3 text-sm">
                  <p className="font-medium text-slate-200">{tool.name}</p>
                  <p className="text-slate-400">{tool.description}</p>
                   {tool.note && <p className="text-xs text-amber-400 mt-1">Note: {tool.note}</p>}
                </div>
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

interface ToolSelectorProps {
  categories: ToolCategory[];
  selectedToolIds: Set<string>;
  onToolToggle: (toolId: string) => void;
  onSelectAll: (ids: Set<string>) => void;
}

export const ToolSelector: React.FC<ToolSelectorProps> = ({ categories, selectedToolIds, onToolToggle, onSelectAll }) => {
  
  const allToolIds = categories.flatMap(c => c.tools.map(t => t.id));
  
  const handleSelectAll = () => {
    onSelectAll(new Set(allToolIds));
  };

  const handleDeselectAll = () => {
    onSelectAll(new Set());
  };

  return (
    <div className="p-6 bg-slate-800/50 border border-slate-700 rounded-xl">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-slate-100">Select Tools</h2>
        <div className="flex space-x-2">
            <button onClick={handleSelectAll} className="text-sm px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded-md transition-colors">Select All</button>
            <button onClick={handleDeselectAll} className="text-sm px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded-md transition-colors">Deselect All</button>
        </div>
      </div>
      <div className="space-y-4">
        {categories.map(category => (
          <AccordionItem
            key={category.name}
            category={category}
            selectedToolIds={selectedToolIds}
            onToolToggle={onToolToggle}
          />
        ))}
      </div>
    </div>
  );
};
