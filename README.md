# IIFL Finance - Policy-Aware Customer Support Agent

A lightweight, policy-aware AI customer support prototype for IIFL Finance. It retrieves answers from financial policy and FAQ documents, provides structured JSON output, handles general conversation/greetings, and automatically escalates queries when confidence is low or information is out-of-scope.

---

## 1. How does your solution work?
Our solution loads markdown policy documents dynamically from `data/policies/` and splits them into section-level chunks on startup. When a customer query is received, it first checks for edge cases (empty input) and general conversation (greetings). Next, it computes TF-IDF vector similarity to retrieve the most relevant policy section. Finally, it generates a response grounded strictly in the retrieved policy text (using Gemini 3.6 Flash LLM or a deterministic grounded generator fallback) and formats the output into a standardized JSON schema containing the query, category, answer, source citation, confidence level, and recommended action (`respond` or `escalate`).

---

## 2. Why did you choose this model / framework / approach?
We chose a lightweight RAG pipeline with TF-IDF cosine similarity and Pydantic structured output instead of a complex vector database or heavy framework. This approach avoids unnecessary infrastructure overhead for small document sets, guarantees sub-second retrieval speed, and ensures strict adherence to the output schema. Using Gemini 3.6 Flash with fallback logic ensures high grounding accuracy and resilience even when offline.

---

## 3. What would you improve before production use?
* **Hybrid Semantic Retrieval:** Upgrade TF-IDF to dense vector embeddings combined with BM25 keyword search for higher semantic accuracy.
* **Persistent Vector Store:** Integrate a lightweight vector store like Qdrant or LanceDB for scaling to thousands of enterprise policy documents.
* **Conversation History & Context:** Add session state tracking to maintain context across multi-turn customer dialogues.
* **Human-in-the-Loop Dashboard:** Build an escalation dashboard for customer service agents to review low-confidence queries and provide feedback.
* **Guardrails & PII Anonymization:** Implement input/output sanitization to mask PII (PAN, Aadhaar, account numbers) before LLM processing.

---

## 4. What is one important security or governance concern in financial services?
A critical security and governance concern is **Data Privacy and PII Protection (Data Sovereignty)**. Transmitting customer sensitive financial records or personal identifiable information (PII) to third-party LLMs without strict masking or enterprise privacy SLAs poses a risk of regulatory non-compliance under RBI and DPDP regulations. Enforcing local PII redaction and strict data retention controls is paramount.

---

## 5. What AI coding tools did you use, and how did you use them?
We utilized **Antigravity AI (powered by Gemini)** as an agentic pair programmer. We used it to design the system architecture, write dynamic document chunking logic, implement Pydantic structured output schemas, craft unit test suites, and draft comprehensive documentation.

---

## Quick Start & Usage

### 1. Prerequisites & Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Setup (Optional for LLM mode)
Set your Google Gemini API key to enable generative LLM grounding (the agent automatically falls back to deterministic grounded synthesis if omitted):
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 3. Run Streamlit Web Application (Interactive UI)
```bash
streamlit run app.py
```

### 4. Run Batch Evaluation via CLI (5 Sample Queries)
```bash
python main.py
```

### 5. Interactive CLI Mode
```bash
python main.py --interactive
```

### 6. Run Unit Tests
```bash
PYTHONPATH=. pytest tests/
```

---

## Example Input & Structured Output

### Example 1: Valid Policy Query
**Input Query:**
> *"What are the foreclosure charges for a personal loan after 1 year?"*

**Output JSON:**
```json
{
  "query": "What are the foreclosure charges for a personal loan after 1 year?",
  "category": "Foreclosure & Prepayment",
  "answer": "Based on Foreclosure Prepayment Policy > 2. Personal Loan Prepayment & Foreclosure Terms:\n- **Lock-in Period:** Minimum 6 mandatory EMI payments must be completed before foreclosure or part-prepayment is allowed.\n- **Foreclosure Charges:**\n  - Foreclosure within 7–12 months: 4.0% of the principal outstanding + GST.\n  - Foreclosure after 12 months: 2.0% of the principal outstanding + GST.",
  "source": "Foreclosure Prepayment Policy > 2. Personal Loan Prepayment & Foreclosure Terms",
  "confidence": "high",
  "action": "respond"
}
```

### Example 2: Out of Scope Query (Escalation)
**Input Query:**
> *"Does IIFL Finance offer loans against cryptocurrency or Bitcoin?"*

**Output JSON:**
```json
{
  "query": "Does IIFL Finance offer loans against cryptocurrency or Bitcoin?",
  "category": "Out of Scope",
  "answer": "I apologize, but IIFL Finance does not offer services related to this request based on our policy documentation. Your query has been escalated to an IIFL representative for further assistance.",
  "source": "N/A",
  "confidence": "low",
  "action": "escalate"
}
```
