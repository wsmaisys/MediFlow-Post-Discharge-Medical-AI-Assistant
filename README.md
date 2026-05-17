# 🩺 MediFlow

Post-discharge medical AI assistant for nephrology-focused patient support.

MediFlow is the patient-facing service in this project. It verifies a bundled demo discharge record, keeps the active patient context inside the current app instance, routes the conversation through focused medical-agent nodes, and uses external tools for grounded answers. General nephrology retrieval is delegated to the separately deployed public Nephrology RAG MCP service.

## 🚀 Live Demo

```text
https://mediflow-ai-medical-assistant-785629432566.us-central1.run.app
```

## 🎯 Purpose

MediFlow helps a discharged patient ask practical questions about:

- discharge instructions
- medications listed in the verified demo discharge record
- diet and fluid restrictions
- follow-up plans
- warning signs and escalation guidance
- general nephrology education

This is a demo medical assistant, not a replacement for a clinician. Patient-specific answers must be grounded in the loaded demo discharge record. General nephrology answers should use the public MCP retrieval service whenever possible.

## 🧭 Service Boundary

MediFlow owns:

- patient-facing chat UI
- demo patient verification
- active patient context and thread state
- clinical safety prompting
- handoff between receptionist, lookup, and clinical nodes
- routing decisions for patient lookup, RAG retrieval, and web search

The public Nephrology RAG MCP service owns:

- FAISS vector store loading
- Mistral embeddings
- nephrology document retrieval
- public MCP-compatible JSON-RPC and SSE responses

MediFlow consumes the MCP service through:

```env
NEPHROLOGY_MCP_URL=https://nephrology-rag-mcp-tool-785629432566.us-central1.run.app/mcp
```

## 🧱 Architecture

```text
Browser UI
  -> FastAPI app.py
  -> LangGraph state machine
     -> receptionist node
     -> patient lookup node
     -> clinical node
        -> query_nephrology_docs via public MCP
        -> search_web via DuckDuckGo
```

State is stored with LangGraph `MemorySaver`. There is no persistent conversation database in the current Cloud Run deployment. A thread can continue only while the same app instance keeps its in-memory state.

## 👤 Demo Patient Verification

The demo record lookup requires:

- patient name
- matching discharge date in `YYYY-MM-DD` format

This is demo verification, not real authentication. Do not use it as production identity proofing.

## 🧠 Agent Behavior

The graph is intentionally split into small responsibilities:

- `receptionist`: greets users, collects verification details, and avoids clinical answers before a patient is loaded
- `lookup`: validates the demo patient record and stores the active patient context
- `clinical`: answers verified patient questions, uses the discharge record for patient-specific claims, and calls tools for broader medical context
- `routing`: keeps handoffs deterministic so patient switching, verification, and clinical answering do not blur together

This separation is meant to reduce hallucination, make handoffs predictable, and keep patient context explicit.

## 🗂️ Key Files

```text
app.py                         FastAPI app and routes
src/chatbot_main.py            LangGraph assembly
src/state_and_graph.py         in-memory state schema and checkpointer
src/routing.py                 deterministic graph routing
src/agents_nodes.py            receptionist, lookup, and clinical nodes
src/tools.py                   web search, MCP RAG client, local demo patient lookup
src/llm_models.py              Mistral chat model setup
data/patients.json             bundled demo discharge records
static/index.html              chat UI
static/patients.html           sanitized demo patient summaries
prompt.txt                     standalone prompt reference
```

## 🔌 API Routes

```text
GET  /                         chat UI
GET  /index.html                chat UI
GET  /patients                  demo patient summary page
GET  /api/patients              sanitized demo patient summaries
GET  /data/patients.json        compatibility summary endpoint
GET  /api/health                health check
GET  /api/threads               explains in-memory thread behavior
POST /api/chat                  JSON chat response
POST /api/chat/stream           SSE-style streaming chat response
GET  /api/documentation         API documentation page
```

## ⚙️ Environment

Create `.env` from `.env.example`:

```env
MISTRAL_API_KEY=your_mistral_api_key
NEPHROLOGY_MCP_URL=https://nephrology-rag-mcp-tool-785629432566.us-central1.run.app/mcp
NEPHROLOGY_MCP_TIMEOUT_SECONDS=30
PORT=5000
ALLOWED_ORIGINS=*
ALLOW_CREDENTIALS=false
LANGSMITH_TRACING=false
LANGSMITH_PROJECT=mediflow
```

## 🧪 Local Run

```bash
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open:

```text
http://localhost:5000
```

## ✅ Testing

```bash
pytest tests -q
```

The tests cover routing, agent-node behavior, API routes, and tool contracts.

## ☁️ Deployment Notes

- Designed for independent auto-deploy from the MediFlow repository.
- Requires `MISTRAL_API_KEY`.
- Uses the public Nephrology MCP endpoint by default unless `NEPHROLOGY_MCP_URL` is overridden.
- Uses in-memory conversation state only.
- Cloud Run restarts, redeploys, and scale-out can reset or split conversation memory.
- Do not add SQLite or filesystem persistence unless the deployment target provides durable storage.

## 🛡️ Safety Notes

- Patient-specific claims must come from the verified demo discharge record.
- General nephrology content should be grounded in MCP retrieval when possible.
- Current or recent medical information should use web search.
- Urgent symptoms should direct users to emergency care or their clinical team.
- This repository is a demo and needs real authentication, audit logging, clinical review, privacy controls, and compliance work before production healthcare use.
