# AgentX Development & Setup Guide (macOS)

This document provides step-by-step instructions for configuring and running the AgentX development environment on macOS.

---

## Prerequisites

1. **Operating System**: macOS (Apple Silicon or Intel)
2. **Python**: Python 3.12+ installed (via Homebrew recommended: `brew install python@3.12`)
3. **Node.js**: v20+ (for Frontend in Phase 8)
4. **Docker & Docker Compose**: Optional for local PostgreSQL and Redis containers
5. **Git**: Installed and initialized

---

## Step-by-Step macOS Setup

### 1. Clone & Enter Repository
```bash
git clone <repo-url> AgentX
cd AgentX
```

### 2. Configure Python 3.12 Virtual Environment
```bash
cd backend
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
```
Edit `.env` to configure your API keys (e.g., `OPENAI_API_KEY`, `TAVILY_API_KEY`) when advancing to LLM and research phases.

### 5. Run Verification Tests
```bash
pytest -v
```

### 6. Start the Backend Development Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Verify Endpoints
- **Root Landing**: [http://localhost:8000/](http://localhost:8000/)
- **Health Check**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
- **Interactive OpenAPI Docs**: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
