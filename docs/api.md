# AgentX REST API Specification

Base URL: `http://localhost:8000/api/v1`

---

## 1. System Endpoints

### `GET /`
Returns service metadata and active environment info.
- **Response**: `200 OK`
```json
{
  "name": "AgentX",
  "version": "0.1.0",
  "description": "Autonomous Multi-Agent Research & Intelligence System",
  "environment": "development",
  "docs_url": "/api/v1/docs",
  "health_check": "/api/v1/health"
}
```

### `GET /api/v1/health`
Health check probe for uptime and subsystem status.
- **Response**: `200 OK`
```json
{
  "status": "healthy",
  "app_name": "AgentX",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-11-01T19:45:11.095148Z",
  "services": {
    "api": { "status": "healthy", "details": "FastAPI server running" },
    "database": { "status": "standby", "details": "Configured, ready for migration" },
    "vector_store": { "status": "standby", "details": "ChromaDB ready" }
  }
}
```

---

## 2. Research Lifecycle Endpoints

### `POST /api/v1/research/plan`
Invokes the Planner Agent to generate a validated structured research plan.
- **Request Body**:
  ```json
  {
    "query": "Research the future of artificial intelligence in healthcare.",
    "depth": "standard",
    "additional_context": "Focus on diagnostic imaging and clinical deployment."
  }
  ```
- **Response**: `200 OK` (Returns `PlanGenerationResponse`)

### `POST /api/v1/research/execute`
Orchestrates Planner and concurrent Researcher agents to collect sources, deduplicate URLs, and extract verifiable findings.
- **Request Body**:
  ```json
  {
    "query": "Research the future of artificial intelligence in healthcare.",
    "depth": "standard"
  }
  ```
- **Response**: `200 OK` (Returns `ResearchExecutionResponse` with `MultiAgentResearchResult`)

---

## 3. Full Research Lifecycle Roadmap (Upcoming Phases)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/research/` | Create a new research workflow session |
| `GET` | `/api/v1/research/` | List historical research tasks with pagination |
| `GET` | `/api/v1/research/{id}` | Get research status, progress, and state summary |
| `DELETE` | `/api/v1/research/{id}` | Delete a research task and associated records |
| `GET` | `/api/v1/research/{id}/report` | Retrieve the generated comprehensive report |
| `GET` | `/api/v1/research/{id}/agents` | Retrieve individual agent execution statuses |
| `GET` | `/api/v1/research/{id}/logs` | Stream or fetch live execution logs |
| `POST` | `/api/v1/research/{id}/approve`| Human-in-the-loop: Approve or edit research plan |
| `POST` | `/api/v1/research/{id}/cancel` | Abort an active research execution workflow |
| `GET` | `/api/v1/research/{id}/export/markdown` | Export report as structured Markdown |
| `GET` | `/api/v1/research/{id}/export/pdf` | Export publication-grade PDF report |

