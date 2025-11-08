import React from 'react';
import { Notebook } from '../types';
import { BookOpenIcon } from './icons/BookOpenIcon';

interface StarterNotebooksProps {
  notebooks: Notebook[];
  selectedNotebookId: string | null;
  onSelectNotebook: (id: string | null) => void;
}

export const StarterNotebooks: React.FC<StarterNotebooksProps> = ({
  notebooks,
  selectedNotebookId,
  onSelectNotebook,
}) => {
  return (
    <div className="p-6 bg-slate-800/50 border border-slate-700 rounded-xl">
      <div className="flex items-center gap-3 mb-4">
        <BookOpenIcon className="w-6 h-6 text-green-400" />
        <h2 className="text-2xl font-bold text-slate-100">Starter Notebooks</h2>
      </div>
      <p className="text-sm text-slate-400 mb-6">
        Choose a pre-built project to get started quickly. Selecting a notebook will disable the tool selector below.
      </p>

      <div className="space-y-3">
        {notebooks.map((notebook) => {
          const isSelected = selectedNotebookId === notebook.id;
          return (
            <button
              key={notebook.id}
              onClick={() => onSelectNotebook(isSelected ? null : notebook.id)}
              className={`w-full text-left p-4 border rounded-lg transition-all duration-200
                ${isSelected 
                  ? 'bg-cyan-900/50 border-cyan-700 ring-2 ring-cyan-500/50' 
                  : 'bg-slate-800 border-slate-700 hover:bg-slate-700/50 hover:border-slate-600'
                }`}
            >
              <h3 className="font-semibold text-slate-100">{notebook.name}</h3>
              <p className="text-sm text-slate-400 mt-1">{notebook.description}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
};
