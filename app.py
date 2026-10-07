import os
import tempfile
import streamlit as st
import pypdf
import chromadb
from chromadb.utils import embedding_functions
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool

# ---------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Credit Underwriting System",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Multi-Agent Credit Underwriting & RAG System")
st.markdown("Powered by **CrewAI**, **Google Gemini**, **ChromaDB**, and **Streamlit**.")

# ---------------------------------------------------------
# 2. SECURE ENVIRONMENT & API KEY CONFIGURATION
# ---------------------------------------------------------

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.sidebar.warning("⚠️ GOOGLE_API_KEY not found in Streamlit Secrets.")
    api_key_input = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
    else:
        os.environ["GOOGLE_API_KEY"] = "placeholder_key"

# ---------------------------------------------------------
# 3. SETUP EPHEMERAL CHROMA_DB POLICY RAG (Cached to prevent resource strain)
# ---------------------------------------------------------
@st.cache_resource
def init_chroma_db():
    client = chromadb.EphemeralClient()
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="enterprise_lending_policies",
        embedding_function=emb_fn
    )
    # Seed institutional lending policies and rules
    collection.add(
        documents=[
            "Rule 101: Minimum DSCR for commercial real estate and logistics loans must be 1.25x.",
            "Rule 102: Maximum Debt-to-Income (DTI) ratio allowed is 43% unless an explicit exception is approved by the Risk Committee.",
            "Rule 103: Minimum credit score for Tier 1 pricing is 720. Scores between 680-719 require manual secondary review.",
            "Rule 104: LTV (Loan-to-Value) ratio cannot exceed 80% for commercial properties without primary mortgage insurance.",
            "Rule 105: Industry-specific liquidity requirement: Logistics and supply chain firms must maintain a quick ratio of at least 1.1x.",
            "Rule 106: Maximum single-borrower exposure limit for commercial credit facilities is $5,000,000 without formal syndicate approval.",
            "Rule 107: Working capital loans must be backed by accounts receivable aging reports showing greater than 90% current status."
        ],
        ids=["policy_1", "policy_2", "policy_3", "policy_4", "policy_5", "policy_6", "policy_7"]
    )
    return collection

policy_collection = init_chroma_db()

@tool("Policy Knowledge Base Search")
def search_lending_policies(query: str) -> str:
    """Searches corporate lending guidelines and policy rules database using semantic search."""
    results = policy_collection.query(query_texts=[query], n_results=3)
    documents = results.get("documents", [[]])[0]
    return "\n".join(documents) if documents else "No matching institutional policy found."

# ---------------------------------------------------------
# 4. DETERMINISTIC TOOLS (PDF Reader, Math Calculator, & Risk Scorer)
# ---------------------------------------------------------
@tool("Deterministic Financial Ratio Calculator")
def calculate_financial_ratios(noi: float, annual_debt_service: float, loan_amount: float, 
                               collateral_value: float, monthly_gross_income: float, 
                               monthly_debt_payments: float, quick_ratio: float) -> str:
    """Computes exact DSCR, DTI, LTV, and validates ratios programmatically using Python logic."""
    try:
        dscr = noi / annual_debt_service if annual_debt_service > 0 else 0.0
        ltv = (loan_amount / collateral_value) * 100 if collateral_value > 0 else 0.0
        dti = (monthly_debt_payments / monthly_gross_income) * 100 if monthly_gross_income > 0 else 0.0
        
        return f"""
        --- DETERMINISTIC CALCULATION RESULTS ---
        - Debt Service Coverage Ratio (DSCR): {dscr:.2f}x
        - Loan-to-Value (LTV) Ratio: {ltv:.2f}%
        - Debt-to-Income (DTI) Ratio: {dti:.2f}%
        - Quick Ratio (Liquidity): {quick_ratio:.2f}x
        -----------------------------------------
        """
    except Exception as e:
        return f"Calculation Error: {str(e)}"

@tool("Deterministic Risk Scorer")
def calculate_risk_score(dscr: float, ltv: float, credit_score: int, quick_ratio: float, loan_amount: float) -> str:
    """Computes a deterministic credit risk score, tier, and compliance flags based on hard financial rules."""
    try:
        risk_flags = []
        score_points = 100

        # 1. Credit Score Evaluation
        if credit_score >= 720:
            credit_tier = "Tier 1 (Low Risk)"
        elif 680 <= credit_score <= 719:
            credit_tier = "Tier 2 (Moderate Risk - Manual Review)"
            score_points -= 15
            risk_flags.append("Credit score requires secondary review (Rule 103).")
        else:
            credit_tier = "Tier 3 (High Risk)"
            score_points -= 35
            risk_flags.append("Credit score below institutional threshold.")

        # 2. DSCR Evaluation
        if dscr >= 1.25:
            dscr_status = "Pass"
        else:
            dscr_status = "Fail"
            score_points -= 25
            risk_flags.append(f"DSCR of {dscr:.2f}x breaches minimum 1.25x requirement (Rule 101).")

        # 3. LTV Evaluation
        if ltv <= 80.0:
            ltv_status = "Pass"
        else:
            ltv_status = "Fail"
            score_points -= 20
            risk_flags.append(f"LTV of {ltv:.2f}% exceeds 80% maximum ceiling (Rule 104).")

        # 4. Exposure Check
        if loan_amount > 5000000:
            score_points -= 15
            risk_flags.append("Single-borrower exposure exceeds $5,000,000 threshold (Rule 106).")

        # Determine Final Quantitative Risk Level
        if score_points >= 85 and not risk_flags:
            risk_level = "LOW RISK"
        elif score_points >= 60:
            risk_level = "MEDIUM RISK (Conditional)"
        else:
            risk_level = "HIGH RISK (Denial Recommended)"

        return f"""
        --- DETERMINISTIC RISK ASSESSMENT ---
        - Final Quantitative Risk Score: {score_points}/100
        - Overall Risk Classification: {risk_level}
        - Principal Credit Tier: {credit_tier}
        - DSCR Status: {dscr_status}
        - LTV Status: {ltv_status}
        - Triggered Risk Flags: {risk_flags if risk_flags else "None"}
        --------------------------------------
        """
    except Exception as e:
        return f"Risk Calculation Error: {str(e)}"

@tool("Read Loan Application PDF")
def read_pdf_application(file_path: str) -> str:
    """Extracts raw text content from a local loan application PDF file path."""
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found on disk."
        reader = pypdf.PdfReader(file_path)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text if text.strip() else "Error: PDF contains no extractable text."
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# ---------------------------------------------------------
# 5. STREAMLIT UI & EXECUTION CONTROLS
# ---------------------------------------------------------
st.markdown("### Step 1: Upload Loan Application Document")
uploaded_file = st.file_uploader("Upload applicant PDF document", type=["pdf"])

target_application_path = "sample_application.pdf"

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        target_application_path = tmp_file.name
    st.success(f"Successfully uploaded: {uploaded_file.name}")
else:
    st.info("No file uploaded. Running with default sample payload if triggered.")

if st.button("🚀 Run Multi-Agent Underwriting Assessment", type="primary"):
    if os.environ.get("GOOGLE_API_KEY") in ["", "placeholder_key"]:
        st.error("Please configure your Google Gemini API Key via Streamlit Secrets or the sidebar.")
    else:
        with st.spinner("Agents are analyzing document, executing deterministic calculations, and auditing policies..."):
            try:
                gemini_llm = LLM(model="gemini/gemini-3.6-flash", temperature=0.1)

                intake_agent = Agent(
                    role="Senior Loan Intake Specialist",
                    goal="Ingest application data and output a structured breakdown of application figures.",
                    backstory="Expert loan processor who handles documentation intake.",
                    tools=[read_pdf_application],
                    llm=gemini_llm,
                    verbose=True
                )

                analyst_agent = Agent(
                    role="Financial Credit Analyst",
                    goal="Use the financial calculator tool to compute exact credit ratios.",
                    backstory="You never calculate numbers manually; you rely strictly on the deterministic python calculator tool.",
                    tools=[calculate_financial_ratios],
                    llm=gemini_llm,
                    verbose=True
                )

                risk_agent = Agent(
                    role="Deterministic Credit Risk Assessor",
                    goal="Evaluate borrower risk profile programmatically using the risk scoring tool.",
                    backstory="You never guess risk scores. You evaluate structured metrics against exact institutional risk matrices using the tool.",
                    tools=[calculate_risk_score],
                    llm=gemini_llm,
                    verbose=True
                )

                policy_agent = Agent(
                    role="Underwriting Policy Compliance Officer",
                    goal="Cross-reference computed metrics against institutional rules via vector search.",
                    backstory="Ensures absolute compliance with bank lending guidelines.",
                    tools=[search_lending_policies],
                    llm=gemini_llm,
                    verbose=True
                )

                decision_agent = Agent(
                    role="Senior Credit Committee Chair",
                    goal="Synthesize results into a finalized executive credit memo.",
                    backstory="Makes the final executive call on credit facilities.",
                    llm=gemini_llm,
                    verbose=True
                )

                task_intake = Task(
                    description=f"Read the application source from '{target_application_path}' using the PDF tool. If unavailable, use this fallback data: [Applicant: Apex Logistics LLC, Loan Amount: $1,200,000, NOI: $180,000, Annual Debt Service: $135,000, Monthly Gross Income: $25,000, Monthly Debt: $9,500, Collateral Value: $1,500,000, Credit Score: 705, Quick Ratio: 1.05].",
                    expected_output="Clean structured summary of application parameters.",
                    agent=intake_agent
                )

                task_analysis = Task(
                    description="Take the financial metrics extracted during intake and invoke the 'Deterministic Financial Ratio Calculator' tool with exact numbers to compute DSCR, LTV, DTI, and Quick Ratio.",
                    expected_output="Exact ratio calculations returned by the tool.",
                    agent=analyst_agent
                )

                task_risk = Task(
                    description="Take the computed financial ratios (DSCR, LTV) and applicant profile parameters, then invoke the 'Deterministic Risk Scorer' tool with exact numbers (dscr, ltv, credit_score, quick_ratio, loan_amount).",
                    expected_output="Programmatically derived risk score, classification, and audit flags.",
                    agent=risk_agent
                )

                task_policy = Task(
                    description="Query the policy knowledge base using the search tool to audit rules 101 through 107 against the applicant's calculated metrics.",
                    expected_output="Policy compliance audit matching applicant metrics against institutional rules.",
                    agent=policy_agent
                )

                task_decision = Task(
                    description="Draft a professional Credit Memo including Executive Summary, Financial Breakdown, Risk Rating, Policy Compliance, and Final Committee Recommendation.",
                    expected_output="Finalized Credit Memo ready for sign-off.",
                    agent=decision_agent
                )

                underwriting_crew = Crew(
                    agents=[intake_agent, analyst_agent, risk_agent, policy_agent, decision_agent],
                    tasks=[task_intake, task_analysis, task_risk, task_policy, task_decision],
                    process=Process.hierarchical,
                    manager_llm=gemini_llm,
                    verbose=True
                )

                result = underwriting_crew.kickoff()

                st.markdown("---")
                st.subheader("📋 Final Executive Credit Memo")
                st.markdown(str(result))

            except Exception as e:
                st.error(f"An error occurred during crew execution: {str(e)}")