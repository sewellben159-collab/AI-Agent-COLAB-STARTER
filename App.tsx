import React, { useState, useCallback, useMemo } from 'react';
import { Header } from './components/Header';
import { ToolSelector } from './components/ToolSelector';
import { OutputPane } from './components/OutputPane';
import { TOOL_CATEGORIES, STARTER_NOTEBOOKS } from './constants';
import { Tool, InsightResult, AnalysisResult, Notebook } from './types';
import { ApiKeyManager } from './components/ApiKeyManager';
import { ToolInsights } from './components/ToolInsights';
import { RepoAnalyzer } from './components/RepoAnalyzer';
import { StarterNotebooks } from './components/StarterNotebooks';
import { getToolInsights, analyzeRepository } from './services/geminiService';

function App() {
  const [selectedToolIds, setSelectedToolIds] = useState<Set<string>>(new Set());
  const [selectedNotebookId, setSelectedNotebookId] = useState<string | null>(null);
  const [apiKeys, setApiKeys] = useState<Record<string, string>>({});
  const [resetKey, setResetKey] = useState(0);

  // State for ToolInsights
  const [insightsQuery, setInsightsQuery] = useState('');
  const [insightsResult, setInsightsResult] = useState<InsightResult | null>(null);
  const [isGeneratingInsights, setIsGeneratingInsights] = useState(false);
  const [insightsError, setInsightsError] = useState<string | null>(null);

  // State for RepoAnalyzer
  const [repoUrl, setRepoUrl] = useState('');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzingRepo, setIsAnalyzingRepo] = useState(false);
  const [repoAnalysisError, setRepoAnalysisError] = useState<string | null>(null);

  const allTools: Tool[] = TOOL_CATEGORIES.flatMap(category => category.tools);

  const handleToolToggle = useCallback((toolId: string) => {
    setSelectedNotebookId(null); // Deselect notebook when a tool is toggled
    setSelectedToolIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(toolId)) {
        newSet.delete(toolId);
      } else {
        newSet.add(toolId);
      }
      return newSet;
    });
  }, []);

  const handleSelectNotebook = useCallback((notebookId: string | null) => {
    if (notebookId) {
      setSelectedToolIds(new Set()); // Deselect all tools when a notebook is selected
      setSelectedNotebookId(notebookId);
    } else {
      setSelectedNotebookId(null);
    }
  }, []);

  const handleApiKeyChange = useCallback((keyVar: string, value: string) => {
    setApiKeys(prev => ({ ...prev, [keyVar]: value }));
  }, []);

  const selectedTools: Tool[] = useMemo(() => allTools.filter(tool => selectedToolIds.has(tool.id)), [selectedToolIds, allTools]);
  const selectedNotebook: Notebook | null = useMemo(() => STARTER_NOTEBOOKS.find(nb => nb.id === selectedNotebookId) || null, [selectedNotebookId]);


  const handleGenerateInsights = useCallback(async () => {
    if (selectedTools.length === 0 || !insightsQuery) return;

    setIsGeneratingInsights(true);
    setInsightsError(null);
    setInsightsResult(null);

    try {
      const result = await getToolInsights(selectedTools, insightsQuery);
      setInsightsResult(result);
    } catch (e: any) {
      setInsightsError(e.message || 'An unknown error occurred.');
    } finally {
      setIsGeneratingInsights(false);
    }
  }, [selectedTools, insightsQuery]);

  const handleAnalyzeRepo = useCallback(async () => {
    if (!repoUrl) return;

    setIsAnalyzingRepo(true);
    setRepoAnalysisError(null);
    setAnalysisResult(null);

    try {
      const result = await analyzeRepository(repoUrl, allTools);
      setAnalysisResult(result);
      
      handleSelectNotebook(null); // Switch back to tool mode

      // Auto-select the recommended tools
      const recommendedToolIds = allTools
        .filter(tool => result.recommendedTools.includes(tool.name))
        .map(tool => tool.id);
      
      setSelectedToolIds(prev => {
        const newSet = new Set(prev);
        recommendedToolIds.forEach(id => newSet.add(id));
        return newSet;
      });

    } catch (e: any) {
      setRepoAnalysisError(e.message || 'An unknown error occurred.');
    } finally {
      setIsAnalyzingRepo(false);
    }
  }, [repoUrl, allTools, handleSelectNotebook]);

  const handleReset = useCallback(() => {
    setSelectedToolIds(new Set());
    setSelectedNotebookId(null);
    setApiKeys({});
    setInsightsQuery('');
    setInsightsResult(null);
    setInsightsError(null);
    setRepoUrl('');
    setAnalysisResult(null);
    setRepoAnalysisError(null);
    setResetKey(prev => prev + 1); // This will force remount of child components with a key
  }, []);


  return (
    <div className="min-h-screen bg-slate-900 text-slate-300">
      <Header onReset={handleReset} />
      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          <div className="flex flex-col gap-8">
            <StarterNotebooks 
              notebooks={STARTER_NOTEBOOKS}
              selectedNotebookId={selectedNotebookId}
              onSelectNotebook={handleSelectNotebook}
            />
            <fieldset disabled={!!selectedNotebookId} className="disabled:opacity-50 disabled:pointer-events-none transition-opacity">
              <div className="flex flex-col gap-8">
                 <RepoAnalyzer
                  url={repoUrl}
                  onUrlChange={setRepoUrl}
                  onAnalyze={handleAnalyzeRepo}
                  result={analysisResult}
                  isLoading={isAnalyzingRepo}
                  error={repoAnalysisError}
                />
                <ToolSelector
                  categories={TOOL_CATEGORIES}
                  selectedToolIds={selectedToolIds}
                  onToolToggle={handleToolToggle}
                  onSelectAll={(ids) => {
                    handleSelectNotebook(null);
                    setSelectedToolIds(ids);
                  }}
                />
              </div>
            </fieldset>

            <ApiKeyManager
              selectedTools={selectedTools}
              selectedNotebook={selectedNotebook}
              apiKeys={apiKeys}
              onApiKeyChange={handleApiKeyChange}
            />

            <fieldset disabled={!!selectedNotebookId} className="disabled:opacity-50 disabled:pointer-events-none transition-opacity">
                <ToolInsights
                  selectedTools={selectedTools}
                  query={insightsQuery}
                  onQueryChange={setInsightsQuery}
                  onGenerate={handleGenerateInsights}
                  result={insightsResult}
                  isLoading={isGeneratingInsights}
                  error={insightsError}
                />
            </fieldset>
          </div>
          <OutputPane 
            key={resetKey} 
            selectedTools={selectedTools} 
            selectedNotebook={selectedNotebook}
            apiKeys={apiKeys} 
          />
        </div>
      </main>
    </div>
  );
}

export default App;
