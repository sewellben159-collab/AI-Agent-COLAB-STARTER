import { GoogleGenAI, Type } from "@google/genai";
import { Tool, InsightResult, AnalysisResult } from "../types";

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  // This is a fallback for development, but the environment must have the key.
  console.warn("API_KEY environment variable not set. Gemini API calls will fail.");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

export const generateStarterCode = async (selectedTools: Tool[], apiKeys: Record<string, string>): Promise<string> => {
  if (!API_KEY) {
    throw new Error("API key is not configured. Please set the API_KEY environment variable.");
  }
  if (selectedTools.length === 0) {
    return Promise.resolve("# Please select at least one tool to generate starter code.");
  }

  const toolNames = selectedTools.map(tool => tool.name).join(', ');
  
  const hasProvidedKeys = Object.values(apiKeys).some(value => value && value.trim() !== '');

  let keyInstructions = `Make sure to handle potential API key requirements by instructing the user to replace placeholders like 'YOUR_API_KEY'. Do not use mock keys.`;

  if (hasProvidedKeys) {
    const keySetupCode = Object.entries(apiKeys)
      .filter(([, value]) => value && value.trim() !== '')
      .map(([key, value]) => `os.environ["${key}"] = "${value}"`)
      .join('\n');
      
    keyInstructions = `The user has provided one or more API keys. At the very beginning of the script, import the 'os' module and set the environment variables for these keys using the exact code snippet below. Do NOT use placeholders for these keys.

Provided keys setup code:
\`\`\`python
import os
${keySetupCode}
\`\`\`

For any other API keys that might be required by the tools but were NOT provided by the user, you should show how to set them with a placeholder, but comment out the line. For example: \`# os.environ["SOME_OTHER_KEY"] = "YOUR_SOME_OTHER_KEY"\`.
`;
  }

  const prompt = `
You are an expert AI engineer specializing in Python and Google Colab. Your task is to generate a starter Python script for a Google Colab notebook.

The script should demonstrate a basic 'hello world' or a simple integration example for the following libraries: ${toolNames}.

${keyInstructions}

Provide the code in a single Python code block, ready to be pasted into a Colab cell. Use comments to explain the purpose of different parts of the code.

If libraries are incompatible or difficult to demonstrate together, prioritize showing them individually within the same script, clearly separated by comments.

Example for a single library (e.g., LangChain) when no key is provided:
\`\`\`python
# This is a demonstration for LangChain
# Make sure you have run !pip install langchain langchain-openai

import os
from langchain_openai import OpenAI

# It is recommended to set the API key as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# If the key is not set, you will get an error.
llm = OpenAI()
prompt = "Hello, LangChain!"
response = llm.invoke(prompt)
print(response)
\`\`\`

Now, generate the script for: ${toolNames}.
`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
    });
    
    // Extract the code from markdown code blocks if present
    const rawText = response.text;
    const pythonCodeMatch = rawText.match(/```python\n([\s\S]*?)```/);
    if (pythonCodeMatch && pythonCodeMatch[1]) {
      return pythonCodeMatch[1].trim();
    }
    // Fallback to the raw text if no markdown block is found
    return rawText.trim();
  } catch (error) {
    console.error("Error generating starter code with Gemini:", error);
    throw new Error("Failed to generate starter code. Please check your API key and network connection.");
  }
};

export const getToolInsights = async (selectedTools: Tool[], query: string): Promise<InsightResult> => {
  if (!API_KEY) {
    throw new Error("API key is not configured. Please set the API_KEY environment variable.");
  }
  if (selectedTools.length === 0) {
    return { text: "Please select at least one tool to get insights.", sources: [] };
  }

  const toolNames = selectedTools.map(tool => tool.name).join(', ');

  const prompt = `
    Based on up-to-date information from the web, answer the following question about these AI agent development tools: ${toolNames}.

    Question: "${query}"

    Provide a concise and informative answer.
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        tools: [{googleSearch: {}}],
      },
    });

    const text = response.text;
    const sources = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];

    const webSources = sources.filter(chunk => chunk.web);

    return { text, sources: webSources };
  } catch (error) {
    console.error("Error getting tool insights with Gemini:", error);
    throw new Error("Failed to get insights. The model may have returned an empty response or an error occurred.");
  }
};

export const analyzeRepository = async (repoUrl: string, allTools: Tool[]): Promise<AnalysisResult> => {
  if (!API_KEY) {
    throw new Error("API key is not configured. Please set the API_KEY environment variable.");
  }
  if (!repoUrl) {
    return Promise.reject(new Error("Repository URL is required."));
  }

  const toolNames = allTools.map(tool => tool.name);

  const prompt = `
    You are an expert AI software engineer. Analyze the software project from this URL: ${repoUrl}

    Based on your analysis of the project's purpose, language, and dependencies from its public information (like its README, description, etc.), please provide the following:
    1. A list of recommended tools from the provided list that would be most helpful for this project. The names in your list must exactly match the names from the provided tool list.
    2. A list of other common libraries, packages, or dependencies that a developer might need for this project, formatted as a markdown list.

    The available tools you can choose from are:
    - ${toolNames.join('\n- ')}

    Your response MUST be a valid JSON object.
  `;
  
  const analysisSchema = {
    type: Type.OBJECT,
    properties: {
      recommendedTools: {
        type: Type.ARRAY,
        description: "An array of tool names from the provided list that are suitable for the repository.",
        items: {
          type: Type.STRING,
        },
      },
      otherSuggestions: {
        type: Type.STRING,
        description: "A markdown-formatted string listing other suggested packages, libraries, or dependencies.",
      },
    },
    required: ["recommendedTools", "otherSuggestions"],
  };

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-pro',
      contents: prompt,
      config: {
        // Using the googleSearch tool is unsupported when a JSON mime type is requested.
        // tools: [{googleSearch: {}}],
        responseMimeType: "application/json",
        responseSchema: analysisSchema,
      },
    });

    const jsonString = response.text.trim();
    if (!jsonString.startsWith('{') || !jsonString.endsWith('}')) {
        console.error("Gemini did not return valid JSON:", jsonString);
        throw new Error("The model did not return a valid JSON response. Please try again.");
    }

    const result: AnalysisResult = JSON.parse(jsonString);
    
    if (!result.recommendedTools || typeof result.otherSuggestions === 'undefined') {
      throw new Error("The model's JSON response was missing required fields.");
    }
    
    return result;

  } catch (error: any) {
    console.error("Error analyzing repository with Gemini:", error);
    if (error.message.includes("JSON")) {
       throw new Error("Failed to parse the analysis from the model. It might be helpful to rephrase your request or try again.");
    }
    throw new Error("Failed to analyze repository. Please check the URL and your network connection.");
  }
};