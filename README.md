# Brain-Heart Deep Research System

**Pure LLM Architecture - Zero Hardcoding, Maximum Flexibility**

Advanced AI research platform where **tools do the actual work** and **LLM agents orchestrate everything dynamically**.

## Key Features

### Brain Agent (Pure Orchestrator)
- **Dynamic Tool Selection**: LLM analyzes query and chooses optimal tools
- **Execution Planning**: Creates multi-step plans adapted to each query
- **Memory Integration**: Learns from previous interactions
- **Model Flexibility**: Use any supported LLM

### Heart Agent (Pure Synthesizer)  
- **Communication Strategy**: LLM determines optimal response approach
- **Style Adaptation**: Dynamically adjusts tone and structure
- **Value Optimization**: Maximizes user value through intelligent synthesis

### Professional Tools
- **Calculator**: Mathematical operations, statistics, financial analysis
- **Web Search**: Real-time information retrieval (multiple providers)
- **RAG System**: Knowledge base retrieval with LLM synthesis

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.example .env
# Edit .env with your API keys (minimum: one LLM provider)
```

### 3. Test System
```bash
python test_system.py
```

### 4. Launch Application
```bash
streamlit run app.py
```

## Budget-Friendly Setup

### Recommended for Cost-Conscious Users:
```bash
# Primary LLM (choose one)
OPENROUTER_API_KEY=sk-or-v1-your_key  # Access to multiple models
GROQ_API_KEY=gsk_your_key             # Fast, affordable

# Cost-effective search
VALUESERP_API_KEY=your_key            # $1 per 1000 searches
SERPER_API_KEY=your_key               # $1 per 1000 searches (backup)

# Free social media
REDDIT_CLIENT_ID=your_reddit_id       # Completely free
YOUTUBE_API_KEY=your_youtube_key      # 10K units/day free
```

### Monthly Cost Estimates:
- **Light use** (1,000 searches): $1-2/month
- **Moderate use** (5,000 searches): $5-10/month  
- **Heavy use** (20,000 searches): $20-40/month

## Example Use Cases

### Mathematical Analysis
```
"Calculate compound interest on $50,000 at 6.5% annually for 15 years"
```

### Research Projects  
```
"Research sustainable packaging trends and market opportunities in Europe"
```

### Strategic Analysis
```
"Analyze the competitive landscape for electric vehicle charging networks"
```

## Model Testing Platform

- **Multi-Provider Support**: OpenAI, Anthropic, OpenRouter, Groq
- **Independent Selection**: Different models for Brain vs Heart agents
- **Temperature Control**: Adjust creativity for each agent
- **Performance Comparison**: Test which models work best

## Extending the System

### Adding New LLM Providers:
1. Add provider detection in `config.py`
2. Implement API client in `llm_client.py`

### Creating New Tools:
1. Inherit from `BaseTool` in `tools.py`
2. Implement async `execute()` method
3. Register in `ToolManager`

## File Structure
```
brain-heart-research-system/
├── core/                   # Core system modules
├── app.py                 # Streamlit interface
├── requirements.txt       # Dependencies
├── .env.example          # Configuration template
└── README.md             # This file
```

## Troubleshooting

**No API Keys Detected:**
- Verify .env file exists and contains valid keys
- Restart application after adding keys

**Tool Failures:**
- Web search requires API key (SERPER_API_KEY or VALUESERP_API_KEY)
- Other tools work without external dependencies

---

*Brain-Heart Deep Research System v2.0 - Pure LLM Architecture with Zero Hardcoding*