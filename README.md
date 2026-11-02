# AgentX

**Autonomous Multi-Agent Research & Intelligence System**  
*Tagline: Multiple Agents. One Intelligent Outcome.*

---

## Overview

AgentX is an autonomous multi-agent platform designed to address complex research queries by distributing tasks among specialized AI agents:
1. **Orchestrator**: Manages state, transitions, and workflow lifecycle.
2. **Planner**: Deconstructs questions into structured execution objectives.
3. **Researcher**: Concurrently discovers reliable sources and collects evidence with Tavily.
4. **Analyst**: Synthesizes data, uncovers patterns, and distinguishes facts from commentary.
5. **Fact Checker**: Validates claims against source evidence with strict verification statuses.
6. **Writer**: Compiles an 11-section professional report with direct citations.
7. **Reviewer**: Evaluates report quality, enforcing QA criteria and feedback loops.

---

## Tech Stack

- **Backend**: Python 3.12+, FastAPI, Pydantic, Uvicorn, SQLAlchemy
- **Multi-Agent**: LangGraph, LangChain, langchain-openai
- **LLM & Search**: OpenAI API (configurable models), Tavily Search API
- **Persistence & Memory**: PostgreSQL, Alembic, ChromaDB, Redis
- **Frontend**: React, TypeScript, Vite, Tailwind CSS (Phase 8)
- **Observability**: Structured Logging, LangSmith

---

## Quick Start (macOS)

```bash
# 1. Setup backend virtual environment
cd backend
python3.12 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Environment configuration
cp .env.example .env

# 4. Run automated test suite
pytest -v

# 5. Launch FastAPI development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health Check: `http://localhost:8000/api/v1/health`
- Plan Generation: `POST http://localhost:8000/api/v1/research/plan`
- Research Execution: `POST http://localhost:8000/api/v1/research/execute`
- Swagger Docs: `http://localhost:8000/api/v1/docs`

---

## Roadmap

- [x] **Phase 1: Foundation (FastAPI, Logging, Configuration, Health Probe)**
- [x] **Phase 2: First AI Agent (Planner Agent & Structured Pydantic Output)**
- [x] **Phase 3: First Multi-Agent Collaboration (Researcher + Tavily + Parallel Execution)**
- [ ] Phase 4: LangGraph Stateful Workflow & Checkpointing
- [ ] Phase 5: Research Intelligence (Analyst & Fact Checker)
- [ ] Phase 6: Report Generation (Writer & Reviewer QA Loop)
- [ ] Phase 7: Persistent Memory (PostgreSQL & ChromaDB)
- [ ] Phase 8: Full-Stack Frontend (React + Vite + Tailwind Dashboard)
- [ ] Phase 9: Advanced Capabilities (Dynamic Agents, Document Intel, Debate)
- [ ] Phase 10: Docker & Production Deployment

