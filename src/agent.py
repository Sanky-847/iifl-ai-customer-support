import os
import re
import json
import logging
from typing import Optional
from dotenv import load_dotenv
from src.models import SupportResponse
from src.retriever import PolicyRetriever

# Automatically load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class CustomerSupportAgent:
    """
    Policy-Aware Customer Support Agent.
    Combines retrieval, grounding, confidence scoring, general conversation awareness,
    and fallback escalation handling.
    """

    def __init__(self, policy_dir: str = "data/policies", api_key: Optional[str] = None):
        self.retriever = PolicyRetriever(policy_dir=policy_dir)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini Client: {e}. Falling back to rule-guided generator.")

    def process_query(self, query: str) -> SupportResponse:
        """Processes a customer query and returns a structured SupportResponse."""

        # 1. Edge Case: Empty / Whitespace Input
        if not query or not query.strip():
            return SupportResponse(
                query=query if query is not None else "",
                category="Error Handling",
                answer="No question provided. Please enter a valid customer query.",
                source="N/A",
                confidence="low",
                action="escalate"
            )

        cleaned_query = query.strip()

        # 2. General Conversation Awareness (Greetings / Chit-chat / Overview requests)
        if self._is_general_conversation(cleaned_query):
            return SupportResponse(
                query=cleaned_query,
                category="General Conversation",
                answer="Hello! I am IIFL Finance's AI Customer Assistant. I can assist you with information on Personal Loans, Gold Loans, and Foreclosure/Prepayment policies. How can I help you today?",
                source="N/A",
                confidence="high",
                action="respond"
            )

        # 3. Explicit Out-of-Scope / Unsupported Keyword Detection
        if self._is_explicitly_out_of_scope(cleaned_query):
            return SupportResponse(
                query=cleaned_query,
                category="Out of Scope",
                answer="I apologize, but IIFL Finance does not offer services related to this request based on our policy documentation. Your query has been escalated to an IIFL representative for further assistance.",
                source="N/A",
                confidence="low",
                action="escalate"
            )

        # 4. Policy Retrieval Step (Minimum threshold 0.15 for meaningful match)
        retrieved_results = self.retriever.retrieve(cleaned_query, top_k=3, score_threshold=0.15)

        # 5. Out-of-Scope / Insufficient Information Check
        if not retrieved_results:
            return SupportResponse(
                query=cleaned_query,
                category="Out of Scope",
                answer="I apologize, but I could not find information regarding this in IIFL's official policy documents. Your query has been escalated to an IIFL representative for further assistance.",
                source="N/A",
                confidence="low",
                action="escalate"
            )

        # Format context for LLM grounding
        top_chunk, top_score = retrieved_results[0]
        context_str = "\n\n".join([
            f"[Source: {chunk.source_reference}]\n{chunk.content}"
            for chunk, score in retrieved_results
        ])

        # 6. Generate Grounded Response (via Gemini LLM or Local Grounded Generator)
        if self.client:
            try:
                return self._generate_llm_response(cleaned_query, context_str, top_chunk.source_reference, top_score)
            except Exception as e:
                logger.warning(f"LLM invocation failed: {e}. Using deterministic grounded response.")

        return self._generate_fallback_grounded_response(cleaned_query, retrieved_results)

    def _is_general_conversation(self, query: str) -> bool:
        """Classifies greetings, pleasantries, and general assistant queries."""
        q = query.lower().strip()

        # Common greeting prefixes or phrases
        greeting_patterns = [
            r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b",
            r"who are you",
            r"what can you do",
            r"what services (do you|can you) (provide|offer)",
            r"how can you help",
            r"^help$"
        ]

        for pattern in greeting_patterns:
            if re.search(pattern, q):
                # If it's a greeting but also contains a specific policy keyword (e.g. "hi, what are personal loan rates?"), let retriever handle it
                policy_keywords = {"rate", "interest", "doc", "document", "foreclosure", "prepayment", "fee", "ltv", "gold", "cibil", "income"}
                words = set(re.findall(r'\w+', q))
                if not words.intersection(policy_keywords):
                    return True

        return False

    def _is_explicitly_out_of_scope(self, query: str) -> bool:
        """Detects explicitly unsupported topics like crypto, stocks, gambling, etc."""
        out_of_scope_keywords = {
            "crypto", "cryptocurrency", "bitcoin", "ethereum", "btc", "eth", "nft",
            "stock market", "trading", "gambling", "lottery"
        }
        words = set(re.findall(r'\w+', query.lower()))
        return bool(words.intersection(out_of_scope_keywords))

    def _generate_llm_response(self, query: str, context: str, primary_source: str, score: float) -> SupportResponse:
        """Invokes Gemini LLM with multi-model fallback and strict grounding."""
        from google.genai import types

        system_instruction = (
            "You are an AI Customer Support Agent for IIFL Finance. "
            "Answer the customer question strictly grounded in the provided policy context below. "
            "If the context does not contain enough information to give a complete answer, state that "
            "and mark action as 'escalate' and confidence as 'low'.\n\n"
            f"POLICY CONTEXT:\n{context}"
        )

        prompt = f"Customer Query: {query}"
        candidate_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash"]

        last_err = None
        for model_name in candidate_models:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=SupportResponse,
                        temperature=0.1
                    )
                )
                parsed_data = json.loads(response.text)
                return SupportResponse(**parsed_data)
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}. Trying next candidate model.")
                last_err = e
                continue

        raise last_err

    def _generate_fallback_grounded_response(self, query: str, retrieved_results) -> SupportResponse:
        """Deterministic response synthesizer used when running without API key."""
        top_chunk, top_score = retrieved_results[0]
        
        # Determine confidence level based on cosine similarity top_score
        if top_score >= 0.35:
            confidence = "high"
            action = "respond"
        elif top_score >= 0.20:
            confidence = "medium"
            action = "respond"
        else:
            confidence = "low"
            action = "escalate"

        # Deduce category from source document
        doc_name = top_chunk.doc_name
        if "Personal" in doc_name:
            category = "Personal Loan Policy"
        elif "Gold" in doc_name:
            category = "Gold Loan FAQ"
        elif "Foreclosure" in doc_name:
            category = "Foreclosure & Prepayment"
        else:
            category = "General Policy"

        answer = f"Based on {top_chunk.source_reference}:\n{top_chunk.content[:300]}..."

        if action == "escalate":
            answer += "\n\n(Note: Information coverage is limited. Escalating to human support for confirmation.)"

        return SupportResponse(
            query=query,
            category=category,
            answer=answer,
            source=top_chunk.source_reference,
            confidence=confidence,
            action=action
        )
