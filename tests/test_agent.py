import pytest
from src.agent import CustomerSupportAgent
from src.retriever import PolicyRetriever


@pytest.fixture
def agent():
    return CustomerSupportAgent(policy_dir="data/policies")


@pytest.fixture
def retriever():
    return PolicyRetriever(policy_dir="data/policies")


def test_retriever_chunk_loading(retriever):
    assert len(retriever.chunks) > 0
    # Check that chunks are properly parsed from markdown files
    doc_names = {c.doc_name for c in retriever.chunks}
    assert "Personal Loan Policy" in doc_names
    assert "Gold Loan Faq" in doc_names
    assert "Foreclosure Prepayment Policy" in doc_names


def test_valid_policy_query(agent):
    query = "What documents are required for a personal loan?"
    response = agent.process_query(query)

    assert response.query == query
    assert response.action in ["respond", "escalate"]
    assert "Personal" in response.category or "Policy" in response.category
    assert response.source != "N/A"
    assert len(response.answer) > 0


def test_out_of_scope_query(agent):
    query = "Do you accept Ethereum or Bitcoin as collateral for loans?"
    response = agent.process_query(query)

    assert response.query == query
    assert response.confidence == "low"
    assert response.action == "escalate"
    assert response.category == "Out of Scope"


def test_general_conversation(agent):
    query = "Hello, good morning!"
    response = agent.process_query(query)

    assert response.query == query
    assert response.category == "General Conversation"
    assert response.action == "respond"
    assert response.source == "N/A"
    assert response.confidence == "high"


def test_empty_input(agent):
    query = "   "
    response = agent.process_query(query)

    assert response.category == "Error Handling"
    assert response.action == "escalate"
    assert response.confidence == "low"
