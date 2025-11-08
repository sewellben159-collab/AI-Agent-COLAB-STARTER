export interface ApiKeyInfo {
  envVar: string;
  label: string;
  placeholder: string;
  docsUrl: string;
}

export interface Tool {
  id: string;
  name: string;
  description: string;
  installCommand: string;
  note?: string;
  apiKeys?: ApiKeyInfo[];
}

export interface ToolCategory {
  name: string;
  tools: Tool[];
}

export interface Notebook {
  id: string;
  name: string;
  description: string;
  content: string;
  apiKeys?: ApiKeyInfo[];
}

export interface GroundingChunkWeb {
  uri: string;
  title: string;
}

export interface GroundingChunk {
  web: GroundingChunkWeb;
}

export interface InsightResult {
  text: string;
  sources: GroundingChunk[];
}

export interface AnalysisResult {
  recommendedTools: string[];
  otherSuggestions: string;
}
