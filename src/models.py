from typing import Literal
from pydantic import BaseModel, Field


class SupportResponse(BaseModel):
    query: str = Field(..., description="The input query received from the customer.")
    category: str = Field(
        ...,
        description="Category of the query (e.g., 'Personal Loan', 'Gold Loan', 'Foreclosure & Prepayment', 'General Conversation', 'Out of Scope', 'Error Handling')."
    )
    answer: str = Field(
        ...,
        description="Detailed answer grounded in policy documents, general response, or escalation statement."
    )
    source: str = Field(
        ...,
        description="The source policy document and section reference, or 'N/A' if not applicable."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Confidence level of the response based on retrieval relevance and policy coverage."
    )
    action: Literal["respond", "escalate"] = Field(
        ...,
        description="Action recommendation: 'respond' to answer directly, or 'escalate' to hand off to human support."
    )
