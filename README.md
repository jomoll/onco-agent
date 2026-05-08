<div align="center">
<h1>
  Onco-Agent: Clinical Question-Answering Agent
</h1>
</div>
<p align="center">
📝 <a href="https://arxiv.org/pdf/2604.24473" target="_blank">Paper</a> • 🌐 <a href="https://jomoll.github.io/onco-agent/" target="_blank">Project</a>
</p>


**Onco-Agent** is a multi-turn retrieval-augmented agent for answering structured clinical questions from patient records. It is the system described in our paper:

> [Agentic clinical reasoning over longitudinal myeloma records: a retrospective evaluation against expert consensus](https://arxiv.org/pdf/2604.24473)

Given a patient's clinical record (discharge letters, radiology reports, lab values, etc.) and a doctor's question, the agent:

1. **Plans** which information to retrieve and which domain skills to apply.
2. **Retrieves** relevant report sections and lab values using specialized retrieval tools with date and type filters.
3. **Applies** domain-specific skills (clinical workflows, evidence ranking, scoring systems).
4. **Produces** a structured answer with inline citations.
  
<table>
<tr>
<td align="center" width="33%">
<video src="https://github.com/user-attachments/assets/d5902b41-c3af-427a-a705-2dd349df6712" controls width="100%"></video>
<br>Single-document lookup
</td>

<td align="center" width="33%">
<video src="https://github.com/user-attachments/assets/e4c06a0a-06fc-4a8e-ac7e-9859a8229ca2" controls width="100%"></video>
<br>Temporal reasoning
</td>

<td align="center" width="33%">
<video src="https://github.com/user-attachments/assets/d7b1bfd7-aea1-43d6-831f-671a7cdd8249" controls width="100%"></video>
<br>Multi-criteria synthesis
</td>
</tr>
</table>



---

## Quick start

### 1. Install dependencies

```bash
conda create -n onco-agent python=3.10
conda activate onco-agent
pip install -r requirements.txt
```

### 2. Configure your LLM endpoint

```bash
cp .env.example .env
# Edit .env and set VLLM_BASE_URL, VLLM_MODEL, and OPENAI_API_KEY
```

The agent is backend-agnostic. Any OpenAI-compatible endpoint works (vLLM, Ollama, Azure OpenAI). Google Gemini via Vertex AI is also supported — set `GEMINI_PROJECT_ID` and authenticate with `gcloud auth application-default login`.

### 3. Build a patient database

Create one JSON file per patient (see [Input format](#input-format)) and run:

```bash
python scripts/build_database.py cases/ --db src/database/synthetic.sqlite
```

### 4. Run the agent

**Single question, single patient:**
```bash
python scripts/run_agent.py \
  --db src/database/synthetic.sqlite \
  --patient-id patient_001 \
  --question "What is the patient's ISS stage at diagnosis?"
```

**Batch: all patients, questions from a file:**
```bash
python scripts/run_agent.py \
  --db src/database/synthetic.sqlite \
  --all-patients \
  --questions-file questions.json \
  --output answers.json
```

**Specific patients from a list file:**
```bash
python scripts/run_agent.py \
  --db src/database/synthetic.sqlite \
  --patient-ids-file patient_ids.txt \
  --questions-file questions.json \
  --output answers.json
```

### 5. (Optional) Start the REST API

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

Set `CLINICAL_AGENT_BACKEND=gemini` (or `vllm`) in `.env` to choose the backend. For Gemini, ensure `GEMINI_PROJECT_ID`, `GEMINI_LOCATION`, `GEMINI_MODEL`, and `GEMINI_ALLOWED_MODELS` are set.

### 6. (Optional) Start the web UI

```bash
cd src/clinical-rag-ui
# Point the UI at your backend (edit src/clinical-rag-ui/.env):
#   AGENT_API_URL=http://<server-ip>:8000
npm install && npm run dev
```

The Vite dev server proxies all `/api/*` requests to `AGENT_API_URL` server-side, so no CORS configuration is needed regardless of which machine the browser connects from.

---

## Input format

Each patient is described by a single JSON file. A directory of such files is passed to `build_database.py`.

```json
{
  "patient_id":    "patient_001",
  "firstname":     "Max",
  "lastname":      "Mustermann",
  "date_of_birth": "1960-03-15",

  "reports": [
    {
      "report_type": "doctor_letter",
      "report_date": "2023-06-15",
      "title":       "Discharge Letter – June 2023",
      "sections": [
        {
          "name":    "Diagnosis",
          "content": "Multiple myeloma IgG kappa, ISS stage II, diagnosed 10/2021."
        },
        {
          "name":    "Current Treatment",
          "content": "VRd induction cycles 1–4 completed. Response: VGPR."
        }
      ]
    },
    {
      "report_type": "radiology",
      "report_date": "2023-05-10",
      "title":       "Whole-Body Low-Dose CT",
      "sections": [
        {
          "name":    "Findings",
          "content": "Lytic lesion L3 stable compared to previous imaging."
        }
      ]
    }
  ],

  "lab_values": [
    {
      "name":            "IgG",
      "date":            "2023-06-14",
      "value":           "18.4",
      "unit":            "g/L",
      "reference_range": "7.0–16.0"
    },
    {
      "name":            "Beta-2-Microglobulin",
      "date":            "2023-06-14",
      "value":           "3.8",
      "unit":            "mg/L"
    }
  ]
}
```

### Valid `report_type` values

| Value | Description |
|---|---|
| `doctor_letter` | Discharge summaries, outpatient letters |
| `consult` | Specialist consultation notes |
| `radiology` | CT, MRI, PET, X-ray, ultrasound reports |
| `pathology` | Histopathology, biopsy reports |
| `tumor_board` | Multidisciplinary tumor board decisions |
| `cardiology` | Echocardiography, cardiology assessments |
| `cytology` | Cytology reports |
| `flow` | Flow cytometry reports |
| `history` | Patient history, anamnesis |

---

## Database schema

The SQLite database created by `build_database.py` contains these tables:

### `patients`
| Column | Type | Description |
|---|---|---|
| `patient_id` | TEXT PK | Unique identifier |
| `firstname` | TEXT | |
| `lastname` | TEXT | |
| `fullname` | TEXT | |
| `dob` | TEXT | Date of birth (YYYY-MM-DD) |

### `reports`
| Column | Type | Description |
|---|---|---|
| `report_id` | TEXT PK | Auto-generated from patient_id + date + title |
| `patient_id` | TEXT FK | |
| `report_type` | TEXT | One of the valid report types above |
| `report_date` | TEXT | YYYY-MM-DD |
| `content` | TEXT | Full markdown content |

### `report_sections` ← *agent retrieves from here*
| Column | Type | Description |
|---|---|---|
| `section_id` | TEXT PK | |
| `report_id` | TEXT FK | |
| `patient_id` | TEXT FK | |
| `section_name` | TEXT | Section heading (shown in citations) |
| `section_content` | TEXT | Section body text |
| `section_order` | INTEGER | Position within the report |
| `report_type` | TEXT | Inherited from parent report |
| `report_date` | TEXT | Inherited from parent report (YYYY-MM-DD) |

### `lab_values` ← *agent retrieves from here*
| Column | Type | Description |
|---|---|---|
| `patient_id` | TEXT FK | |
| `canonical_key` | TEXT | Normalised lab name (lowercase, ASCII) |
| `mapped_name` | TEXT | Display name as provided |
| `date` | TEXT | YYYY-MM-DD |
| `time` | TEXT | HH:MM (optional) |
| `value` | TEXT | Raw value string (may contain `<`, `>`) |
| `value_num` | REAL | Parsed numeric value (if convertible) |
| `unit` | TEXT | |
| `ref_range` | TEXT | Reference range string |

---

## Questions file format

```json
[
  {
    "id": "q01",
    "question": "What is the ISS stage at diagnosis?",
    "answer_schema": "score_value_plus_date_and_source"
  },
  {
    "id": "q02",
    "question": "Which therapy lines has the patient received?"
  }
]
```

The `answer_schema` field is optional. It guides the agent toward a specific structured output format. Available schemas are defined in `src/skills/schema/`.

---

## Answer format

Answers follow a structured format controlled by `answer_schema`. Examples:

```
Answer: ISS II | 15.06.2023 | Calculated
Reasoning: β2-Microglobulin 3.8 mg/L and Albumin 4.1 g/dL satisfy ISS II criteria [ctx:report_001].

Answer: Status=Documented || VRd | 10.2021 - 02.2022; Dara-VRd | 03.2022 - 09.2022
Reasoning: Four cycles of VRd documented in discharge letter from 15.02.2022 [ctx:report_002].
```

---

## Architecture

```
scripts/
  build_database.py   — Ingests patient JSON files → SQLite
  run_agent.py        — CLI entry point for batch/interactive runs

src/
  agent_base.py       — Multi-turn DSPy orchestration (plan → retrieve → answer)
  agent_tools.py      — RAG tools: retrieve_reports, retrieve_lab_values + scoring calculators
  agent_vllm.py       — OpenAI-compatible LLM backend
  agent_gemini.py     — Google Gemini backend
  api_server.py       — FastAPI REST endpoint
  signatures.py       — DSPy signature definitions
  toolkit.py          — Tool primitives (llama_index wrapper)
  llm_output.py       — Output sanitisation
  lab_catalog_resolver.py  — Lab name normalisation
  report_type_synonyms.py  — Report type alias mapping

  skills/
    policy_engine.py  — Evidence ranking, temporal authority, contradiction resolution
    registry.py       — Skill loading and prompt assembly
    policy/           — YAML policy rules
    schema/           — Answer schema definitions per question type
    style/            — Output style instructions
    workflows/        — Multi-step clinical reasoning workflows

  clinical-rag-ui/    — React/Vite web frontend

tests/
  test_agents_base.py        — Agent orchestration unit tests
  test_agents_tools.py       — Tool execution tests
  test_agents_integration.py — End-to-end integration tests (requires database + LLM)
  test_lab_catalog_resolver.py
  test_ipssr_hctci.py        — Domain scoring tests
```

---

## Running tests

```bash
# Unit tests (no LLM or database required)
pytest tests/ -m "not integration"

# Integration tests (requires populated database and running LLM endpoint)
pytest tests/ -m integration
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `AGENT_API_URL` | — | URL of the backend API server used by the Vite dev proxy (e.g. `http://10.x.x.x:8000`). Set in `src/clinical-rag-ui/.env`. Not exposed to the browser. |
| `VLLM_BASE_URL` | — | OpenAI-compatible endpoint URL |
| `VLLM_MODEL` | — | Model name |
| `OPENAI_API_KEY` | — | API key |
| `VLLM_TEMPERATURE` | — | Sampling temperature (omit for model default) |
| `VLLM_MAX_COMPLETION_TOKENS` | — | Max tokens per completion |
| `VLLM_CONTEXT_TOKENS` | `120000` | Context window limit (tokens) |
| `GEMINI_PROJECT_ID` | — | GCP project ID for Vertex AI Gemini backend |
| `GEMINI_LOCATION` | `us-central1` | Vertex AI region |
| `GEMINI_MODEL` | `gemini-3.1-pro-preview` | Gemini model name |
| `GEMINI_ALLOWED_MODELS` | — | Comma-separated list of models exposed in the UI |
| `CLINICAL_AGENT_BACKEND` | `vllm` | Backend selection: `vllm` or `gemini` |
| `DB_PATH` | `src/database/synthetic.sqlite` | SQLite database path |
| `ANSWER_LANGUAGE` | `English` | Language for generated answers |
| `MAX_TOOL_ROUNDS` | `8` | Maximum retrieval rounds per question |
| `USE_HYBRID_RETRIEVAL` | `0` | Enable hybrid BM25+embedding retrieval (`1` to enable) |

---

## Citation

```bibtex
@article{moll2026agentic,
  title={Agentic clinical reasoning over longitudinal myeloma records: a retrospective evaluation against expert consensus},
  author={Moll, Johannes and L{\"u}bberstedt, Jannik and Nuernbergk, Christoph and Stroh, Jacob and Mertens, Luisa and Purcarea, Anna and Zirn, Christopher and Benchaaben, Zeineb and Drexel, Fabian and H{\"a}ntze, Hartmut and others},
  journal={arXiv preprint arXiv:2604.24473},
  year={2026}
}
```
