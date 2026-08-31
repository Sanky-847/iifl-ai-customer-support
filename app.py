import os
import json
import streamlit as st
from dotenv import load_dotenv
from src.agent import CustomerSupportAgent
from src.retriever import PolicyRetriever

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    page_title="IIFL Customer Support Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-respond {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-escalate {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-high {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #F3F4F6;
        color: #4B5563;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .supported-box {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
    }
    .format-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=64)
    st.title("IIFL AI Agent Settings")
    
    st.markdown("### 🔑 LLM Configuration")
    # Support environment variables, .env, and Streamlit secrets
    secret_key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            secret_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    server_api_key = os.getenv("GEMINI_API_KEY") or secret_key or os.getenv("GOOGLE_API_KEY", "")
    
    api_key_input = st.text_input(
        "Custom Gemini API Key (Optional)",
        value="",
        type="password",
        placeholder="Enter key to override server key..." if server_api_key else "Enter Gemini API Key...",
        help="The server already has a secure Gemini API key configured. You can optionally enter your own key here to override it."
    )
    
    effective_api_key = api_key_input.strip() if api_key_input.strip() else server_api_key

    if effective_api_key:
        st.success("🟢 LLM Mode: Gemini Active")
        if not api_key_input and server_api_key:
            st.caption("🔒 Using secure server-side key (hidden from clients)")
    else:
        st.info("🟡 Mode: Local Grounded Fallback (Offline)")

    st.markdown("---")
    st.markdown("### 📋 About")
    st.markdown(
        "A policy-aware AI customer support prototype for **IIFL Finance**, "
        "providing grounded answers on Personal Loans, Gold Loans, and Foreclosure terms with automated query escalation."
    )
    st.markdown("---")
    st.caption("IIFL Finance AI Engineer Round 1 Prototype")


# Initialize Agent
policies_dir = "data/policies"
os.makedirs(policies_dir, exist_ok=True)
agent = CustomerSupportAgent(policy_dir=policies_dir, api_key=effective_api_key if effective_api_key else None)

# Main Title Area
st.markdown('<div class="main-header">🏦 IIFL Finance Support Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Policy-Aware Customer Query Resolution & Automated Escalation Engine</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Query & Chat Tester", 
    "➕ Upload Custom Policies", 
    "⚡ 1-Click Batch Evaluation", 
    "📚 Policy Documents Explorer"
])


# TAB 1: Query Tester
with tab1:
    st.subheader("Ask a Customer Question")

    # Sample query selector for quick testing
    sample_options = [
        "-- Select a quick test query or type below --",
        "What are the foreclosure charges for a personal loan after 1 year?",
        "What income documents do I need to submit for a salaried personal loan?",
        "What is the minimum gold purity required for an IIFL Gold Loan?",
        "Does IIFL Finance offer loans against cryptocurrency or Bitcoin?",
        "Hi there! Good morning, can you tell me what services you provide?",
        "Can I part-prepay my gold loan before the tenure ends?",
        "Looking for charges info if I pay in let's say 18 months time my loan"
    ]
    
    selected_sample = st.selectbox("Quick Preset Questions:", sample_options)
    default_text = "" if selected_sample == sample_options[0] else selected_sample

    user_query = st.text_area(
        "Customer Query Input:",
        value=default_text,
        placeholder="Type a customer question here (e.g. 'What is the processing fee for a gold loan?')",
        height=100
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        submit_btn = st.button("🚀 Process Query", type="primary", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("Clear", use_container_width=False)

    if submit_btn and user_query:
        with st.spinner("Analyzing policy documents and generating grounded response..."):
            response = agent.process_query(user_query)
            res_dict = response.model_dump()

        st.markdown("---")
        st.subheader("Result & Structured Output")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("#### 💡 Agent Answer")
            
            # Action and Confidence Badges
            action_class = "badge-respond" if res_dict["action"] == "respond" else "badge-escalate"
            action_icon = "✅" if res_dict["action"] == "respond" else "⚠️"
            conf_class = f"badge-{res_dict['confidence']}"

            st.markdown(f"""
            <div style="margin-bottom: 12px;">
                <span class="{action_class}">{action_icon} Action: {res_dict['action'].upper()}</span> &nbsp;
                <span class="{conf_class}">Confidence: {res_dict['confidence'].capitalize()}</span> &nbsp;
                <span class="badge-high">Category: {res_dict['category']}</span>
            </div>
            """, unsafe_allow_html=True)

            st.info(res_dict["answer"])
            st.markdown(f"**📖 Source Citation:** `{res_dict['source']}`")

        with col_right:
            st.markdown("#### 📦 Structured JSON Output")
            st.json(res_dict)


# TAB 2: Upload Custom Policies
with tab2:
    st.subheader("Upload Custom Policy Documents")
    st.markdown("Add your own company policy circulars, FAQs, or rate sheets to the agent's knowledge base in real-time.")

    # Allowed Document Formats Box
    st.markdown("""
    <div class="supported-box">
        <h4 style="margin: 0 0 10px 0; color: #166534;">📋 Allowed Policy Document Types</h4>
        <div>
            <span class="format-badge">📄 Markdown (.md, .txt)</span>
            <span class="format-badge">📑 PDF Documents (.pdf)</span>
            <span class="format-badge">📊 PowerPoint Slides (.pptx, .ppt)</span>
            <span class="format-badge">📦 Structured JSON (.json)</span>
        </div>
        <p style="margin: 10px 0 0 0; color: #374151; font-size: 0.88rem;">
            Uploaded files will be immediately parsed, chunked, and indexed by the Policy Retriever. You can then immediately switch to the <b>Query & Chat Tester</b> tab and ask questions grounded in your uploaded documents!
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose policy document files to upload:",
        type=["pdf", "pptx", "ppt", "json", "md", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Save & Index Documents", type="primary"):
            added_count = 0
            for uploaded_file in uploaded_files:
                save_path = os.path.join(policies_dir, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                added_count += 1

            st.success(f"✅ Successfully uploaded and indexed {added_count} document(s)! You can now ask questions about them in the Query tab.")
            st.rerun()


# TAB 3: Batch Evaluation
with tab3:
    st.subheader("Run Batch Evaluation (5 Required Assignment Test Cases)")
    st.markdown("Evaluate all sample customer queries covering valid policy queries, greetings, out-of-scope topics, and edge cases.")

    if st.button("▶️ Run Full Batch Evaluation", type="primary"):
        queries_file = "data/sample_queries.json"
        if os.path.exists(queries_file):
            with open(queries_file, "r", encoding="utf-8") as f:
                sample_queries = json.load(f)

            with st.spinner("Executing batch evaluation across all test cases..."):
                for item in sample_queries:
                    q_id = item.get("id")
                    query_text = item.get("query", "")
                    desc = item.get("description", "")

                    res = agent.process_query(query_text)
                    res_dict = res.model_dump()

                    action_color = "green" if res_dict["action"] == "respond" else "red"
                    with st.expander(f"**Test Case #{q_id}**: {desc} — *[{res_dict['action'].upper()}]*", expanded=True):
                        st.markdown(f"**Query:** `{query_text if query_text else '<EMPTY INPUT>'}`")
                        c1, c2 = st.columns([3, 2])
                        with c1:
                            st.markdown(f"**Category:** {res_dict['category']}")
                            st.markdown(f"**Action:** `{res_dict['action']}` | **Confidence:** `{res_dict['confidence']}`")
                            st.markdown(f"**Answer:** {res_dict['answer']}")
                            st.markdown(f"**Source:** `{res_dict['source']}`")
                        with c2:
                            st.json(res_dict)
            st.success(f"Batch evaluation successfully completed ({len(sample_queries)} test cases processed).")
        else:
            st.error(f"Sample queries file '{queries_file}' not found.")


# TAB 4: Policy Documents Explorer
with tab4:
    st.subheader("Currently Indexed Policy & FAQ Documents")
    st.markdown(f"Total chunks indexed in memory: **{len(agent.retriever.chunks)} sections**")

    if os.path.exists(policies_dir):
        policy_files = [f for f in os.listdir(policies_dir) if any(f.lower().endswith(ext) for ext in agent.retriever.ALLOWED_EXTENSIONS)]
        for pfile in sorted(policy_files):
            filepath = os.path.join(policies_dir, pfile)
            ext = os.path.splitext(pfile)[1].lower()
            
            icon = "📄"
            if ext == ".pdf":
                icon = "📑"
            elif ext in [".pptx", ".ppt"]:
                icon = "📊"
            elif ext == ".json":
                icon = "📦"

            file_size_kb = os.path.getsize(filepath) / 1024
            
            with st.expander(f"{icon} {pfile} ({file_size_kb:.1f} KB)", expanded=False):
                if ext in [".md", ".txt", ".json"]:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        st.code(f.read(), language="markdown" if ext != ".json" else "json")
                else:
                    st.info(f"Binary document `{pfile}` parsed and indexed across pages/slides in retriever.")
    else:
        st.warning("Policies directory not found.")
