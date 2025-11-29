# 🏥 MediFlow

> **Intelligent Post-Discharge Medical AI Assistant**  
> Powered by LangChain, LangGraph, and Mistral AI

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Core Components](#-core-components)
- [Configuration](#-configuration)
- [Usage & Examples](#-usage--examples)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Performance](#-performance)
- [Security](#-security)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Resources](#-resources)

---

## 🎯 Overview

MediFlow is a **production-ready, multi-agent AI chatbot** designed to provide intelligent medical guidance and personalized support to patients after hospital discharge. It combines cutting-edge LLM technologies with evidence-based medical knowledge retrieval.

### What Makes MediFlow Special?

✨ **Intelligent Routing** - Separate receptionist and clinical agents for optimal user experience  
✨ **Knowledge Integration** - FAISS-based RAG pipeline for medical expertise  
✨ **Real-time Information** - Web search fallback for up-to-date medical information  
✨ **Complete Observability** - Full tracing via LangSmith for transparency  
✨ **Privacy-Focused** - PHI/PII aware architecture  
✨ **Production-Ready** - Docker support, comprehensive testing, streaming responses

---

## ✨ Key Features

| Feature                   | Description                                                             |
| ------------------------- | ----------------------------------------------------------------------- |
| 🤖 **Multi-Agent System** | Specialized agents for greeting, data retrieval, and clinical reasoning |
| 📚 **RAG Pipeline**       | FAISS vector store for accurate medical knowledge retrieval             |
| 🌐 **Web Search**         | Real-time DuckDuckGo integration with intelligent fallback              |
| 📊 **Full Tracing**       | Complete LangSmith integration for debugging and monitoring             |
| ⚡ **Token Streaming**    | Real-time response streaming for responsive user experience             |
| 🔐 **Security First**     | PHI/PII redaction, environment-based configuration                      |
| 🧪 **95%+ Test Coverage** | Comprehensive unit and integration tests                                |
| 🐳 **Docker Ready**       | Production deployment with containerization                             |
| 📱 **Modern Web UI**      | Interactive chat interface with patient selector                        |
| ♻️ **Async Support**      | High-performance async/sync bridge architecture                         |

---

## 🚀 Quick Start

### Prerequisites

```
✓ Python 3.9+
✓ Mistral API key (get it at https://console.mistral.ai)
✓ Git
✓ (Optional) LangSmith API key for tracing
```

### 5-Minute Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/MediFlow.git
cd MediFlow

# 2. Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cat > .env << EOF
MISTRAL_API_KEY=your_api_key_here
DEBUG=false
LOG_LEVEL=INFO
EOF

# 5. Verify installation
pytest tests/ -v

# 6. Run the application
python app.py
# Open http://localhost:8000 in your browser
```

---

## 🏗️ Architecture

### System Overview

```
┌────────────────────────────────────────────┐
│          Web Interface (UI)                │
│   FastAPI + HTML/JavaScript                │
└─────────────────────┬──────────────────────┘
                      │
┌─────────────────────▼──────────────────────┐
│      LangGraph Orchestrator                 │
│  • State management                        │
│  • Agent routing & control                 │
│  • Tool invocation                         │
│  • LangSmith tracing                       │
└─────────────────────┬──────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────▼─────┐  ┌────▼─────┐ ┌────▼──────┐
   │Receptionist  │  │Patient    │ │Clinical    │
   │Agent         │  │Data Node  │ │Agent       │
   └────┬─────┘  └────┬─────┘ └────┬──────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────▼────┐   ┌────▼────┐  ┌────▼────┐
   │Web Search│   │RAG Tool │  │Patient   │
   │(DuckDuckGo)  │(FAISS)   │  │Database  │
   └─────────┘   └─────────┘  └──────────┘
```

### Data Flow

1. **User Message** → Receptionist Agent
2. **Intent Classification** → Route to Clinical Agent
3. **Context Loading** → Patient Data Node retrieves medical history
4. **Tool Selection** → Clinical agent decides: RAG search, web search, or direct response
5. **Response Generation** → Synthesis with citations and streaming
6. **Observation** → Complete trace in LangSmith

## Project Structure

```
MediFlow/
├── 📄 app.py                          # FastAPI entry point (localhost:8000)
├── 📄 requirements.txt                # Python dependencies
├── 🐳 Dockerfile                      # Container configuration
├── 📝 README.md                       # This file
│
├── 📁 src/                            # Core application code
│   ├── llm_models.py                  # Mistral LLM initialization
│   ├── tools.py                       # Tool definitions (RAG, search, patient)
│   ├── agents_nodes.py                # Agent implementations
│   ├── routing.py                     # Conditional routing logic
│   ├── state_and_graph.py             # LangGraph state & execution graph
│   ├── chatbot_main.py                # CLI orchestrator
│   ├── utilities.py                   # Helper functions
│   ├── utils_async.py                 # Async/sync bridge
│   └── diag_flow.py                   # Diagnostic flow handling
│
├── 📁 tests/                          # Test suite (95%+ coverage)
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_agents_nodes.py           # Agent logic tests
│   ├── test_chatbot_main.py           # Integration tests
│   ├── test_llm_models.py             # LLM tests
│   ├── test_routing.py                # Routing tests
│   ├── test_state_and_graph.py        # State management tests
│   ├── test_tools.py                  # Tool execution tests
│   ├── test_utilities.py              # Utility tests
│   ├── test_utils_async.py            # Async tests
│   └── test_vector_store.py           # RAG tests
│
├── 📁 data/                           # Patient datasets
│   └── patients.json                  # Sample patient records
│
├── 📁 static/                         # Web assets
│   ├── index.html                     # Chat interface
│   ├── patients.html                  # Patient selector
│   └── api_documentation.html         # API reference
│
├── 📁 docs/                           # Documentation
│   ├── plan.md                        # Architecture & design decisions
│   └── README_UI.md                   # Frontend documentation
│
└── 📄 promt.txt                       # Agent system prompts
```

---

## 🔧 Core Components

### 1. LLM Models (`src/llm_models.py`)

**Technology**: Mistral AI  
**Models Used**:

- `mistral-small-latest` - Fast responses (Receptionist)
- `mistral-large-latest` - Complex reasoning (Clinical)

**Configuration**:

```python
Receptionist: Temperature 0.3 (deterministic)
Clinical:    Temperature 0.7 (nuanced reasoning)
```

### 2. Multi-Agent System (`src/agents_nodes.py`)

#### Receptionist Agent

- Warm greeting and authentication
- Intent classification (medical vs. administrative)
- Patient ID extraction
- Smooth handoff to clinical agent

#### Patient Data Node

- Retrieves patient demographics
- Extracts medications and allergies
- Builds enriched context
- Stores in state for personalization

#### Clinical Agent

- Medical reasoning engine
- Orchestrates tool calls (RAG, web search)
- Generates citations and explanations
- Supports multi-turn conversations

### 3. Tools (`src/tools.py`)

| Tool                 | Purpose               | Source         | Output             |
| -------------------- | --------------------- | -------------- | ------------------ |
| `search_web`         | Real-time information | DuckDuckGo API | Formatted results  |
| `query_medical_docs` | Medical knowledge     | FAISS index    | Top-k documents    |
| `patient_data_tool`  | Patient context       | patients.json  | Demographics, meds |

### 4. Graph Architecture (`src/state_and_graph.py`)

- **State Management**: TypedDict with `add_messages` reducer
- **Recursion Limit**: 25 (prevents infinite loops)
- **Persistence**: MemorySaver for conversation history
- **Message Accumulation**: Proper context preservation

### 5. API Endpoints (`app.py`)

```
POST   /api/chat              Send message → get response
POST   /api/chat/stream       Streaming response (SSE)
GET    /api/patients          List available patients
GET    /api/patients/{id}     Get patient details
GET    /                      Serve web UI
GET    /docs                  Interactive API documentation
```

---

## ⚙️ Configuration

### Environment Variables

**Required**:

```bash
MISTRAL_API_KEY=sk-...              # Mistral AI API key
```

**Optional**:

```bash
LANGSMITH_API_KEY=ls_...            # LangSmith tracing
LANGSMITH_PROJECT=mediflow          # Project name
LANGSMITH_TRACING=true              # Enable tracing
DEBUG=false                         # Debug mode
LOG_LEVEL=INFO                      # Logging level
```

### Performance Tuning

**Model Selection** (`src/llm_models.py`):

```python
# Fast responses
"mistral-small-latest"

# Balanced (recommended)
"mistral-medium-latest"

# Complex reasoning
"mistral-large-latest"
```

**Recursion Limit** (`chatbot_main.py`):

```python
config = {"recursion_limit": 25}  # Max tool calls
```

---

## 💻 Usage & Examples

### Option 1: Web UI (Recommended)

```bash
python app.py
# Open http://localhost:8000
```

### Option 2: Command Line

```bash
python src/chatbot_main.py
# Interactive CLI prompt
```

### Option 3: Python API

```python
from src.chatbot_main import create_graph
from src.state_and_graph import ChatState

graph = create_graph()
state = ChatState(messages=[...])
result = graph.invoke(state)
print(result["messages"][-1].content)
```

### Option 4: Docker

```bash
docker build -t mediflow .
docker run -p 8000:8000 --env-file .env mediflow
```

---

## 🧪 Testing

### Run All Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Generate coverage report
open htmlcov/index.html
```

### Run Specific Tests

```bash
# Agent tests only
pytest tests/test_agents_nodes.py -v

# Skip integration tests
pytest tests/ -v -m "not integration"

# Run with detailed output
pytest tests/test_chatbot_main.py -vv -s
```

### Test Structure

- **Unit Tests**: Individual component functionality
- **Integration Tests**: Component interaction
- **Fixtures**: Mocked LLM, database, APIs
- **Coverage**: 95%+ code coverage

---

## 🚢 Deployment

### Local Deployment

```bash
python app.py
# http://localhost:8000
```

### Docker Deployment

```bash
# Build image
docker build -t mediflow:latest .

# Run container
docker run \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  mediflow:latest
```

### Production Checklist

- [ ] Set `DEBUG=false`
- [ ] Use strong API keys
- [ ] Enable LangSmith tracing
- [ ] Configure CORS properly
- [ ] Set up HTTPS/TLS
- [ ] Use PostgreSQL instead of SQLite
- [ ] Configure log aggregation
- [ ] Set up monitoring alerts

---

## ⚡ Performance

### Response Times (Typical)

| Scenario              | Time     | Details            |
| --------------------- | -------- | ------------------ |
| Receptionist greeting | 0.5-1.5s | No tool calls      |
| Clinical response     | 2-4s     | RAG retrieval only |
| Web search            | 5-8s     | Tool execution     |
| Full pipeline         | 8-12s    | All agents + tools |

### Benchmarks

**Environment**: Python 3.11, Mistral API, FAISS index

| Operation       | Time   | Notes               |
| --------------- | ------ | ------------------- |
| LLM inference   | 2-3s   | Includes streaming  |
| FAISS search    | 50ms   | Local vector store  |
| Web search      | 1-2s   | DuckDuckGo API      |
| Tool execution  | <500ms | Negligible overhead |
| Graph traversal | <100ms | State management    |

### Optimization Tips

1. **Model Selection**: Use `mistral-small-latest` for speed
2. **Caching**: Enable LangChain caching for repeated queries
3. **Batch Processing**: Process multiple patients concurrently
4. **Vector Store**: Pre-build FAISS index offline

---

## 🔐 Security & Privacy

### PHI/PII Handling

✅ Patient IDs used in logs (not names)  
✅ Configurable LangSmith trace redaction  
✅ Local-first data storage  
✅ Environment-based secrets (no hardcoding)  
✅ Input validation and sanitization

### Access Control

- Demo: Local access only (localhost:8000)
- Production: OAuth2/LDAP recommended
- Patient Data: Isolated per patient ID
- Audit Trail: Complete LangSmith traces

### Best Practices

```bash
# ✓ DO: Use environment variables
export MISTRAL_API_KEY="sk-..."

# ✗ DON'T: Hardcode secrets
api_key = "sk-..."  # Never do this

# ✓ DO: Rotate keys regularly
# ✓ DO: Enable LangSmith for audit trails
# ✓ DO: Use HTTPS in production
# ✓ DO: Monitor for suspicious patterns
```

---

## 🐛 Troubleshooting

### "MISTRAL_API_KEY not found"

```bash
# Solution: Create .env file
echo "MISTRAL_API_KEY=sk-..." > .env

# Or set environment variable
export MISTRAL_API_KEY="sk-..."
```

### "Tool execution failed"

```bash
# Solution: Enable LangSmith tracing
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="ls_..."

# Check traces at: https://smith.langchain.com
```

### Slow response times

```bash
# Check model selection
grep "mistral-" src/llm_models.py

# Monitor API latency
# Use mistral-small-latest for speed
```

### Streaming not working

```bash
# Browser support check: SSE compatible
# Firewall check: Not buffering responses
# Headers check: Content-Type: text/event-stream
```

### Port already in use

```bash
# Windows
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000

# Change port in app.py or environment
```

---

## 🤝 Contributing

We welcome contributions! Here's how:

```bash
# 1. Fork repository
# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Make changes & test
pytest tests/ -v

# 4. Commit with clear message
git commit -m "feat: Add amazing feature"

# 5. Push and create Pull Request
git push origin feature/amazing-feature
```

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install black pylint pytest-cov

# Format code
black src/ tests/

# Run linter
pylint src/

# Run tests
pytest tests/ -v --cov=src
```

---

## 📚 Resources

### Documentation

- 🏗️ [Architecture Details](docs/plan.md) - Deep dive into design decisions
- 🎨 [UI Documentation](docs/README_UI.md) - Frontend guide
- 📖 [API Reference](http://localhost:8000/docs) - Interactive Swagger docs

### Framework Documentation

- [LangChain](https://python.langchain.com) - LLM orchestration
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Multi-agent framework
- [Mistral AI](https://docs.mistral.ai) - LLM documentation
- [FastAPI](https://fastapi.tiangolo.com) - Web framework

### External Tools

- [LangSmith](https://smith.langchain.com) - LLM observability
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [DuckDuckGo API](https://duckduckgo.com/api) - Web search

---

## 📊 Roadmap

### ✅ Completed

- [x] Multi-agent architecture
- [x] RAG pipeline with FAISS
- [x] Web search integration
- [x] Streaming responses
- [x] LangSmith tracing
- [x] Comprehensive testing
- [x] Docker support
- [x] Security considerations

### 🚧 In Development

- [ ] User authentication
- [ ] PostgreSQL migration
- [ ] Session management
- [ ] Conversation export

### 🚀 Planned

- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Sentiment analysis
- [ ] EHR integration
- [ ] Mobile app

---

## 📞 Support & Contact

### Getting Help

- 🐛 **Bug Reports**: [Open an Issue](https://github.com/yourusername/MediFlow/issues)
- 💬 **Questions**: [Discussions](https://github.com/yourusername/MediFlow/discussions)
- 📖 **Documentation**: See `docs/plan.md`
- 🔍 **Examples**: Check `tests/` and test cases

### Enable Debug Mode

```bash
# Set debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with verbose output
python app.py
```

---

## 📄 License

MIT License - See LICENSE file for details

**Project Scope**: Created as part of an AI/GenAI assignment demonstrating advanced LLM applications in healthcare. Feel free to use as a reference for your projects.

---

## ⭐ Acknowledgments

Built with cutting-edge open-source technologies:

- [**LangChain**](https://python.langchain.com) - LLM orchestration framework
- [**LangGraph**](https://langchain-ai.github.io/langgraph/) - Stateful multi-agent system
- [**Mistral AI**](https://mistral.ai) - Advanced language model
- [**FastAPI**](https://fastapi.tiangolo.com) - Modern web framework
- [**FAISS**](https://github.com/facebookresearch/faiss) - Vector similarity search
- [**LangSmith**](https://smith.langchain.com) - LLM observability platform

---

<div align="center">

**Made with ❤️ for healthcare AI**

If you find MediFlow helpful, please give it a ⭐!

[GitHub](https://github.com/yourusername/MediFlow) • [Issues](https://github.com/yourusername/MediFlow/issues) • [Discussions](https://github.com/yourusername/MediFlow/discussions)

</div>
