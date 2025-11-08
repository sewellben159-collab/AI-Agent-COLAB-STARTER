import { ToolCategory, Notebook } from './types';

export const TOOL_CATEGORIES: ToolCategory[] = [
  {
    name: 'Multi-Agent Frameworks',
    tools: [
      { id: 'langchain', name: 'LangChain', description: 'A widely adopted framework for building applications with LLMs.', installCommand: '!pip install langchain langchain-openai', apiKeys: [{ envVar: 'OPENAI_API_KEY', label: 'OpenAI API Key', placeholder: 'sk-xxxxxxxx', docsUrl: 'https://platform.openai.com/api-keys' }] },
      { id: 'autogen', name: 'AutoGen', description: 'A Microsoft-backed framework for creating applications where multiple agents work together.', installCommand: '!pip install pyautogen', apiKeys: [{ envVar: 'OPENAI_API_KEY', label: 'OpenAI API Key', placeholder: 'sk-xxxxxxxx', docsUrl: 'https://platform.openai.com/api-keys' }] },
      { id: 'crewai', name: 'CrewAI', description: 'Orchestrates role-playing autonomous AI agents to foster collaborative intelligence.', installCommand: '!pip install crewai', apiKeys: [{ envVar: 'OPENAI_API_KEY', label: 'OpenAI API Key', placeholder: 'sk-xxxxxxxx', docsUrl: 'https://platform.openai.com/api-keys' }] },
      { id: 'langflow', name: 'Langflow', description: 'A visual tool for building AI workflows with a drag-and-drop interface.', installCommand: '!pip install langflow' },
      { id: 'autogpt', name: 'AutoGPT', description: 'Build and manage AI agents for complex, continuous workflows.', installCommand: '!git clone https://github.com/Significant-Gravitas/AutoGPT.git', note: 'Requires manual setup after cloning. See repo for details.' },
      { id: 'portia', name: 'Portia', description: 'Developer-focused framework for building predictable and stateful agentic workflows.', installCommand: '!pip install portia-sdk' },
    ],
  },
  {
    name: 'Browser Automation',
    tools: [
      { id: 'playwright', name: 'Playwright', description: 'Framework for web testing and automation across Chromium, Firefox, and WebKit.', installCommand: '!pip install playwright\n!playwright install' },
      { id: 'firecrawl', name: 'Firecrawl', description: 'Turns entire websites into clean markdown or structured data with a single API call.', installCommand: '!pip install firecrawl-py', apiKeys: [{ envVar: 'FIRECRAWL_API_KEY', label: 'Firecrawl API Key', placeholder: 'fc-xxxxxxxx', docsUrl: 'https://firecrawl.dev' }] },
      { id: 'puppeteer', name: 'Puppeteer (Pyppeteer)', description: 'A Python port of the popular Chrome automation library.', installCommand: '!pip install pyppeteer' },
      { id: 'stagehand', name: 'Stagehand', description: 'Mix natural language commands with traditional code for browser automation.', installCommand: '!pip install stagehand' },
    ],
  },
  {
    name: 'Computer Use',
    tools: [
        { id: 'open-interpreter', name: 'Open Interpreter', description: 'Lets an AI agent execute code locally on your computer.', installCommand: '!pip install open-interpreter', note: 'Requires user confirmation for code execution.' },
        { id: 'self-operating-computer', name: 'Self-Operating Computer', description: 'Allows multimodal models to control the mouse and keyboard.', installCommand: '!git clone https://github.com/OthersideAI/self-operating-computer.git', note: 'Experimental and requires specific environment setup.' },
    ]
  },
  {
    name: 'Vertical Agents (Coding, Research)',
    tools: [
      { id: 'aider', name: 'Aider', description: 'An AI pair programmer that works directly in your terminal.', installCommand: '!pip install aider-chat', apiKeys: [{ envVar: 'OPENAI_API_KEY', label: 'OpenAI API Key', placeholder: 'sk-xxxxxxxx', docsUrl: 'https://platform.openai.com/api-keys' }] },
      { id: 'vanna', name: 'Vanna', description: 'Connects to your SQL database for natural language querying.', installCommand: '!pip install vanna' },
      { id: 'gpt-researcher', name: 'GPT Researcher', description: 'Autonomous agent for in-depth research and report generation.', installCommand: '!pip install gpt-researcher', apiKeys: [{ envVar: 'OPENAI_API_KEY', label: 'OpenAI API Key', placeholder: 'sk-xxxxxxxx', docsUrl: 'https://platform.openai.com/api-keys' }, { envVar: 'TAVILY_API_KEY', label: 'Tavily API Key', placeholder: 'tvly-xxxxxxxx', docsUrl: 'https://app.tavily.com/' }] },
      { id: 'screenshot-to-code', name: 'Screenshot-to-code', description: 'Turns visual designs into clean HTML/JS/CSS code.', installCommand: '!pip install screenshot-to-code' },
    ],
  },
  {
    name: 'Memory',
    tools: [
      { id: 'mem0', name: 'Mem0', description: 'An intelligent memory layer that allows AI agents to remember and learn.', installCommand: '!pip install mem0ai' },
      { id: 'langmem', name: 'LangMem', description: 'Helps agents learn from interactions and maintain memory across sessions.', installCommand: '!pip install langmem' },
    ],
  },
  {
    name: 'Evaluation & Monitoring',
    tools: [
      { id: 'langfuse', name: 'Langfuse', description: 'Open-source LLM engineering platform for observability and metrics.', installCommand: '!pip install langfuse', apiKeys: [{ envVar: 'LANGFUSE_PUBLIC_KEY', label: 'Langfuse Public Key', placeholder: 'pk-lf-xxxxxxxx', docsUrl: 'https://langfuse.com/docs/get-started' }, { envVar: 'LANGFUSE_SECRET_KEY', label: 'Langfuse Secret Key', placeholder: 'sk-lf-xxxxxxxx', docsUrl: 'https://langfuse.com/docs/get-started' }] },
      { id: 'openllmetry', name: 'OpenLLMetry', description: 'Extensions on OpenTelemetry for complete LLM application observability.', installCommand: '!pip install openllmetry-sdk' },
      { id: 'agentops', name: 'AgentOps', description: 'Python SDK for monitoring AI agents and tracking costs.', installCommand: '!pip install agentops', apiKeys: [{ envVar: 'AGENTOPS_API_KEY', label: 'AgentOps API Key', placeholder: 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx', docsUrl: 'https://agentops.ai/' }] },
      { id: 'giskard', name: 'Giskard', description: 'Automatically detects performance, bias, and security issues in AI apps.', installCommand: '!pip install giskard' },
    ],
  },
   {
    name: 'Document Processing',
    tools: [
      { id: 'paddleocr', name: 'PaddleOCR', description: 'Multilingual OCR and document parsing toolkit.', installCommand: '!pip install paddlepaddle paddleocr' },
      { id: 'docling', name: 'Docling', description: 'Simplifies document processing and integrates with generative AI tools.', installCommand: '!pip install docling' },
    ],
  },
  {
    name: 'Voice & Speech',
    tools: [
      { id: 'whisper', name: 'OpenAI Whisper', description: 'General-purpose speech recognition for multilingual transcription and translation.', installCommand: '!pip install -U openai-whisper' },
      { id: 'pipecat', name: 'Pipecat', description: 'Python framework for building real-time voice and multimodal conversational AI.', installCommand: '!pip install pipecat-ai' },
      { id: 'pyannote-audio', name: 'Pyannote Audio', description: 'A pipeline that identifies different speakers in an audio stream.', installCommand: '!pip install pyannote.audio' },
    ],
  },
];

export const STARTER_NOTEBOOKS: Notebook[] = [
  {
    id: 'multi-llm-comparison',
    name: 'Multi-LLM Comparison & Analysis',
    description: 'Compare outputs from 8 different LLMs for the same prompt, analyze their similarity, and train a simple TensorFlow model to learn response patterns.',
    apiKeys: [
      { envVar: 'HF_TOKEN', label: 'Hugging Face Token', placeholder: 'hf_xxxxxxxxxxxxxxxxxx', docsUrl: 'https://huggingface.co/settings/tokens' }
    ],
    content: `#@title Cell 1: Install required libraries
!pip install transformers torch tensorflow huggingface_hub requests numpy pandas scikit-learn matplotlib seaborn

#@title Cell 2: Import libraries
import requests
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import InferenceClient  # For easier API access
import warnings
warnings.filterwarnings('ignore')

#@title Introduction & Overview
# # Multi-LLM Comparison with TensorFlow Learning Neural Network
#
# This notebook demonstrates:
# - Querying multiple large language models (LLMs) via Hugging Face Inference API for the same input prompt.
# - Comparing their outputs (e.g., via TF-IDF similarity and qualitative analysis).
# - Training a simple TensorFlow LSTM-based neural network to learn patterns from natural language inputs and LLM outputs, focusing on "coding manifestations" (e.g., generating code snippets or predicting output styles).
#
# **Note:** 
# - This uses Hugging Face's free Inference API, which has rate limits. For production, use paid endpoints or local hosting.
# - Models listed: Llama 3.1, Mixtral 8x22B, Qwen2, DeepSeek R1, Gemma-2-9b-it, Falcon 180B, Phi-4, Mistral-Large-Instruct-2407.
# - Not all models may be directly available via free API; some fallbacks to similar available models if needed (e.g., 'meta-llama/Llama-3.1-8B-Instruct' for Llama 3.1).
# - The TF NN learns to predict/imitate output styles by treating sequences as time series of tokens (simplified for demo).
# - Input example: A coding-related prompt to elicit "manifestations" (e.g., code generation).

#@title Cell 3: Define models and HF API setup
# List of models (using available HF repo IDs for Instruct variants where possible)
MODELS = {
    'Llama-3.1': 'meta-llama/Llama-3.1-8B-Instruct',  # Smaller variant for API feasibility
    'Mixtral-8x22B': 'mistralai/Mixtral-8x22B-Instruct-v0.1',
    'Qwen2': 'Qwen/Qwen2-7B-Instruct',  # Smaller for demo
    'DeepSeek-R1': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',  # Fallback if needed
    'Gemma-2-9b-it': 'google/gemma-2-9b-it',
    'Falcon-180B': 'tiiuae/falcon-180B-chat',  # Huge, may timeout on free API
    'Phi-4': 'microsoft/Phi-3-mini-4k-instruct',  # Phi-3 as proxy for Phi-4 (assuming similar)
    'Mistral-Large-Instruct-2407': 'mistralai/Mistral-Large-Instruct-2407'
}

# Hugging Face token (get from https://huggingface.co/settings/tokens)
HF_TOKEN = "your_hf_token_here"  # Replace with your actual token for authentication

client = InferenceClient(token=HF_TOKEN)

#@title Cell 4: Define function to query a single model
def query_model(model_id, prompt, max_tokens=200):
    try:
        # Use text-generation pipeline via client
        response = client.text_generation(
            prompt,
            model=model_id,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True
        )
        return response
    except Exception as e:
        print(f"Error querying {model_id}: {e}")
        return f"Error: Could not generate output for {model_id}"

# Example input prompt (natural language for coding task)
INPUT_PROMPT = """
Write a Python function to calculate the Fibonacci sequence up to n terms, using recursion, and explain the code in natural language.
"""
print("Using prompt:", INPUT_PROMPT)

#@title Cell 5: Run all models simultaneously on the input
outputs = {}
for name, model_id in MODELS.items():
    print(f"Querying {name}...")
    output = query_model(model_id, INPUT_PROMPT)
    outputs[name] = output
    print(f"{name} output: {str(output)[:100]}...")  # Truncate for display

# Save outputs to DataFrame for comparison
df_outputs = pd.DataFrame(list(outputs.items()), columns=['Model', 'Output'])
df_outputs.to_csv('llm_outputs.csv', index=False)
print(df_outputs)

#@title Cell 6: Compare outputs (qualitative and quantitative)
# TF-IDF for similarity matrix
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df_outputs['Output'].apply(lambda x: str(x)[:500]))  # Truncate for efficiency

similarity_matrix = cosine_similarity(tfidf_matrix)
sns.heatmap(similarity_matrix, xticklabels=df_outputs['Model'], yticklabels=df_outputs['Model'], annot=True, cmap='coolwarm')
plt.title('LLM Output Similarity (Cosine TF-IDF)')
plt.show()

# Print average similarity to input
input_tfidf = vectorizer.transform([INPUT_PROMPT])
input_similarities = cosine_similarity(input_tfidf, tfidf_matrix)[0]
df_outputs['Similarity_to_Input'] = input_similarities
print(df_outputs[['Model', 'Similarity_to_Input']])

#@title TensorFlow NN: Learning from LLM Outputs (Overview)
# # TensorFlow Neural Network: Learning from LLM Outputs
#
# We train a simple LSTM model to learn "manifestations" from the data:
# - **Input:** Tokenized natural language prompt + LLM output (concatenated as sequence).
# - **Target:** Predict next tokens in a coding-style output (seq2seq simplified to next-token prediction).
# - This mimics learning coding patterns and natural language styles from LLM outputs.
# - Dataset: Augmented from our single prompt (in practice, use more data).

#@title Cell 7: Prepare data for TF NN (simple tokenization and sequencing)
# For demo, create synthetic dataset from outputs + variations
all_texts = [INPUT_PROMPT] + [str(out) for out in outputs.values()]
# Simple tokenization (use Tokenizer in full setup)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(all_texts)

# Create input-output pairs for next-token prediction
max_len = 100
X, y = [], []
for seq in sequences:
    for i in range(1, len(seq)):
        n_gram = seq[:i+1]
        if len(n_gram) >= max_len:
            n_gram = n_gram[-max_len:]
        X.append(n_gram[:-1])
        y.append(n_gram[-1])

X = pad_sequences(X, maxlen=max_len-1, padding='pre')
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vocab_size = len(tokenizer.word_index) + 1
print(f"Vocab size: {vocab_size}, Train samples: {len(X_train)}")

#@title Cell 8: Build and train TensorFlow LSTM model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len-1),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

#@title Cell 9: Plot training history and generate sample prediction
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

# Generate sample "learned" output (seed with input tokens)
seed_text = INPUT_PROMPT
seed_seq = tokenizer.texts_to_sequences([seed_text])[0][:max_len-1]
seed_seq = pad_sequences([seed_seq], maxlen=max_len-1, padding='pre')

predicted = []
for _ in range(50):  # Generate 50 tokens
    pred = model.predict(seed_seq, verbose=0)
    next_token = np.argmax(pred[0])
    predicted.append(next_token)
    seed_seq = np.append(seed_seq[:, 1:], [[next_token]], axis=1)

predicted_text = tokenizer.sequences_to_texts([predicted])[0]
print("Learned manifestation (predicted coding/natural language continuation):")
print(predicted_text)

#@title Summary
# # Summary
# - **LLM Comparison:** Outputs from 8 models processed; similarities visualized.
# - **TF Learning:** The LSTM learned basic patterns from inputs/outputs, generating a continuation mimicking coding explanations.
# - **Extensions:** Add more data, fine-tune embeddings (e.g., with CodeBERT), or use RLHF for better "manifestation" learning.
# - Run in Colab with GPU for faster training. Replace HF_TOKEN and handle API limits.
`
  },
];
