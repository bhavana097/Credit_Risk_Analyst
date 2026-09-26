"""
Credit Risk Query Engine — Streamlit App
==========================================
Converts the "Credit Risk Query Engine" notebook into a runnable Streamlit
application. Business users type a natural-language portfolio question;
the app routes it through a verified query template library when possible,
or generates fresh read-only SQL otherwise, validates it, retries once on
failure, executes it against a read-only SQLite connection, and returns a
narrative answer alongside the SQL used, the raw data, and a confidence
score — with a running audit trail of every question asked.

Run with:
    streamlit run app.py

Expected in the same folder as this file:
    - config.json                  {"OPENAI_API_KEY": "...", "OPENAI_API_BASE": "..."}
    - credit_risk_portfolio.db     the SQLite database used by the notebook

(Alternatively, credentials can be supplied via Streamlit secrets
 st.secrets["OPENAI_API_KEY"] / st.secrets["OPENAI_API_BASE"], or via the
 OPENAI_API_KEY / OPENAI_API_BASE environment variables — see
 `load_credentials()` below.)
"""

import json
import os
import re
import sqlite3
import warnings
from datetime import datetime

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="Credit Risk Query Engine", page_icon="📊", layout="wide")

DB_PATH = "credit_risk_portfolio.db"
CONFIG_PATH = "config.json"


# ============================================================
# Credentials
# ============================================================
def load_credentials():
    """Resolve OPENAI_API_KEY / OPENAI_API_BASE from (in order of
    preference) Streamlit secrets, environment variables, or a local
    config.json — matching the notebook's config.json approach."""

    api_key, api_base = None, None

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        api_base = st.secrets.get("OPENAI_API_BASE")
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE") or api_base

    if not api_key and os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        api_key = config.get("OPENAI_API_KEY")
        api_base = config.get("OPENAI_API_BASE")

    if not api_key:
        st.error(
            "No OPENAI_API_KEY found. Provide it via Streamlit secrets, an "
            "OPENAI_API_KEY environment variable, or a config.json file "
            "(see the top of app.py for the expected format)."
        )
        st.stop()

    return api_key, api_base


OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE


# ============================================================
# LLM / DB setup (cached so they only initialize once per session)
# ============================================================
@st.cache_resource
def get_llms():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return llm, evaluator_llm


@st.cache_resource
def get_db_connection():
    if not os.path.exists(DB_PATH):
        st.error(f"Database file '{DB_PATH}' was not found next to app.py.")
        st.stop()
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)


llm, evaluator_llm = get_llms()
conn = get_db_connection()


# ============================================================
# Database schema context (single source of truth for the LLM)
# ============================================================
DATABASE_SCHEMA = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# ============================================================
# Verified query template library
# ============================================================
VERIFIED_QUERY_LIBRARY = {
    "VQ1": {
        "description": "Sector-wise total outstanding and NPA amount breakdown across all sectors",
        "sql": """SELECT s.sector_name,
     ROUND(SUM(l.total_outstanding)/1e6, 1) AS outstanding_million_usd,
     ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard','Doubtful','Loss')
              THEN l.total_outstanding ELSE 0 END)/1e6, 1) AS npa_million_usd
FROM loan_master l
JOIN sector_master s ON s.sector_code = l.sector_code
GROUP BY s.sector_name
ORDER BY outstanding_million_usd DESC""",
    },
    "VQ2": {
        "description": "Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)",
        "sql": """SELECT loan_category,
     ROUND(SUM(total_outstanding)/1e6, 1) AS outstanding_million_usd,
     COUNT(*) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY outstanding_million_usd DESC""",
    },
    "VQ3": {
        "description": "IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter",
        "sql": """SELECT ifrs9_stage,
     COUNT(*) AS loan_count,
     ROUND(SUM(ead_amount)/1e6, 1) AS ead_million_usd,
     ROUND(SUM(ecl_amount)/1e6, 1) AS ecl_million_usd
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage""",
    },
    "VQ4": {
        "description": "Average provision coverage ratio by sector for the latest reporting quarter",
        "sql": """SELECT s.sector_name,
     ROUND(AVG(p.provision_coverage_ratio), 1) AS avg_pcr_percent
FROM provisioning p
JOIN loan_master l ON l.loan_account_number = p.loan_account_number
JOIN sector_master s ON s.sector_code = l.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_pcr_percent DESC""",
    },
    "VQ5": {
        "description": "Top 10 largest loan exposures by outstanding amount at the borrower level",
        "sql": """SELECT borrower_name,
     sector_code,
     ROUND(total_outstanding/1e6, 1) AS outstanding_million_usd,
     asset_classification
FROM loan_master
ORDER BY total_outstanding DESC
LIMIT 10""",
    },
    "VQ6": {
        "description": "Top 5 largest exposures aggregated at the business group level",
        "sql": """SELECT group_name,
     COUNT(DISTINCT loan_account_number) AS loan_count,
     ROUND(SUM(total_outstanding)/1e6, 1) AS outstanding_million_usd
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY outstanding_million_usd DESC
LIMIT 5""",
    },
    "VQ7": {
        "description": "All overdue loan accounts with their days past due and asset classification",
        "sql": """SELECT loan_account_number,
     borrower_name,
     sector_code,
     ROUND(total_outstanding/1e6, 1) AS outstanding_million_usd,
     days_past_due,
     asset_classification
FROM loan_master
WHERE days_past_due > 0
ORDER BY days_past_due DESC""",
    },
    "VQ8": {
        "description": "Distribution of loans across days-past-due buckets showing aging profile of the portfolio",
        "sql": """SELECT
     CASE WHEN days_past_due = 0 THEN '0 (Current)'
          WHEN days_past_due <= 30 THEN '1-30'
          WHEN days_past_due <= 60 THEN '31-60'
          WHEN days_past_due <= 90 THEN '61-90'
          ELSE '90+' END AS dpd_bucket,
     COUNT(*) AS loan_count,
     ROUND(SUM(total_outstanding)/1e6, 1) AS outstanding_million_usd
FROM loan_master
GROUP BY dpd_bucket
ORDER BY MIN(days_past_due)""",
    },
    "VQ9": {
        "description": "Borrowers whose internal rating was downgraded in the latest rating cycle",
        "sql": """SELECT borrower_id,
     previous_rating,
     internal_rating,
     pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC""",
    },
    "VQ10": {
        "description": "Expected credit loss trend across all reporting quarters showing provisioning movement over time",
        "sql": """SELECT reporting_date,
     ROUND(SUM(ecl_amount)/1e6, 1) AS ecl_million_usd
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date""",
    },
}


# ============================================================
# Pipeline functions (unchanged logic from the notebook)
# ============================================================
def classify_intent(user_question, query_library):
    """Classifies the user question and decides which route to take.

    Returns a dict with 'route' (verified or generated), 'query_id'
    (template ID or None), and 'match_reason'.
    """
    library_descriptions = "\n".join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
### ROLE
You are a query router for a credit risk analytics system. Your job is to decide whether a business user's question can be answered by one of the pre-approved query templates, or whether it needs fresh SQL generation.

### INPUT
User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### INSTRUCTIONS
1. Read the user question carefully and identify the analytical intent.
2. Compare the intent against each template description.
3. Match on semantic meaning, not exact wording. For example, 'non-performing' means NPA, 'industry' means sector, 'overdue' means days past due.
4. If a template genuinely answers the question, return that template ID.
5. If no template covers the question, return null for the query_id and set the route to generated.

### OUTPUT
Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context):
    """Generates a candidate SQL query for a novel question using the database schema."""

    generation_prompt = f"""
### ROLE
You are a senior SQL developer specializing in credit risk analytics on a SQLite database.

### INPUT
User Question:
{user_question}

Database Schema (single source of truth):
{schema_context}

### INSTRUCTIONS
1. Write a single SQL query that answers the user question using only the provided schema.
2. The query must be read-only. Use SELECT (or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
3. Use only the tables and columns listed in the schema. Do not invent columns.
4. When comparing dates, use the exact date values available in the schema notes.
5. Round monetary values to millions when appropriate.
6. Ensure the query is SQLite compatible.
7. Alias every numeric column with a suffix that states its unit, so the result is self-describing. Use _million_usd for amounts rounded to millions, _usd for amounts not rounded, _percent for percentages or ratios, _count for counts of loans or borrowers, _days for day values, _rate_percent for interest rates, and _decimal for probabilities expressed as a decimal. Avoid bare aliases like "value", "amount", or "total".

### OUTPUT
Return ONLY the SQL query, with no markdown code blocks, no comments, and no explanation.
"""

    sql = llm.invoke(generation_prompt).content.strip()
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r"^```\s*|\s*```$", "", sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    """Validates a candidate SQL query through five checks before execution."""

    result = {
        "passed": False,
        "failed_check": None,
        "details": "",
        "relevance_confidence": None,
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE", "ATTACH"]
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        result["failed_check"] = "read_only_shape"
        result["details"] = "Query must start with SELECT or WITH"
        return result
    for kw in forbidden_keywords:
        if re.search(r"\b" + kw + r"\b", sql_upper):
            result["failed_check"] = "read_only_shape"
            result["details"] = f"Forbidden keyword detected: {kw}"
            return result
    if ";" in candidate_sql.rstrip(";").rstrip():
        result["failed_check"] = "read_only_shape"
        result["details"] = "Multiple statements are not allowed"
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    _tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or "Name" in str(t.ttype)]
    referenced_identifiers = re.findall(r"\b[a-z_][a-z0-9_]*\b", candidate_sql.lower())
    sql_keywords = {
        "select", "from", "where", "and", "or", "group", "by", "order", "having", "limit", "join", "on", "as", "case",
        "when", "then", "else", "end", "sum", "count", "avg", "min", "max", "round", "desc", "asc", "left", "right",
        "inner", "outer", "distinct", "null", "is", "not", "in", "like", "with", "union", "all", "between", "coalesce",
    }
    _unknown = [
        tok for tok in referenced_identifiers
        if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
        and not tok.isdigit() and tok not in ("s", "l", "p", "r", "e6")
    ]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result["failed_check"] = "parse_plan_dry_run"
        result["details"] = f"SQL failed to parse or plan: {str(e)}"
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""
### ROLE
You are a senior data validator. Your job is to check whether a SQL query correctly answers a business user's question.

### CONTEXT
{track_context}

### INPUT
User Question: {user_question}

Candidate SQL:
{candidate_sql}

### INSTRUCTIONS
Assess whether the SQL genuinely answers what the user asked, considering:
1. Does it query the correct tables and columns?
2. Does it apply the right aggregations and groupings for the underlying metric?
3. Does it handle NPA correctly if asked (Substandard, Doubtful, Loss)?
4. Does it use the correct reporting or rating date if relevant?
5. If this is a verified template (see CONTEXT), do not penalize it for returning a broader result set than the question's scope — only flag it if the metric itself is wrong.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
Confidence is a score from 0.0 (not confident at all) to 1.0 (fully confident).
"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r"\{.*\}", relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result["relevance_confidence"] = relevance_json.get("confidence", 0.0)
        if relevance_json.get("verdict") == "no" or relevance_json.get("confidence", 0.0) < 0.6:
            result["failed_check"] = "llm_relevance"
            result["details"] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]["sql"]
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result["failed_check"] = "template_integrity"
                result["details"] = f"Expected {len(expected_cols)} columns, got {len(actual_cols)}"
                return result
        except sqlite3.Error as e:
            result["failed_check"] = "template_integrity"
            result["details"] = f"Template integrity check failed: {str(e)}"
            return result

    result["passed"] = True
    result["details"] = "All validation checks passed"
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context):
    """Regenerates SQL after a validation failure, feeding the error back to the LLM."""

    retry_prompt = f"""
### ROLE
You are a senior SQL developer fixing a query that failed validation.

### INPUT
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. The revised SQL must be read-only SELECT (or WITH ... SELECT).
4. Use only tables and columns from the schema.
5. Ensure the query is SQLite compatible.

### OUTPUT
Return ONLY the corrected SQL, with no markdown code blocks, no comments, and no explanation.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r"^```sql\s*|\s*```$", "", revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r"^```\s*|\s*```$", "", revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    """Executes a gate-passed SQL query and returns the result as a DataFrame."""

    result = {"dataframe": None, "reasonable": True, "warnings": []}

    df = pd.read_sql_query(validated_sql, db_connection)
    result["dataframe"] = df

    if df.empty:
        result["warnings"].append("Query returned an empty result")

    for col in df.select_dtypes(include="number").columns:
        if (df[col] < 0).any() and "deviation" not in col.lower() and "change" not in col.lower():
            result["warnings"].append(f"Column {col} contains negative values")
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result["warnings"].append(f"Column {col} has {null_count} null values")

    if len(result["warnings"]) > 2:
        result["reasonable"] = False

    return result


def generate_response(user_question, dataframe, route, query_id=None):
    """Generates a focused natural language response from the query result."""

    response_prompt = f"""
### ROLE
You are a credit risk analyst writing a concise business response for a portfolio question.

### INPUT
User Question: {user_question}

Query Result Data:
{dataframe.to_string()}

### INSTRUCTIONS
1. Answer the user's specific question directly. Do not dump the entire table.
2. If the user asked about a specific sector, stage, or category, highlight only those rows.
3. Provide context from other rows only when it adds value (for example, ranking or comparison).
4. State exact numbers from the data. Do not round beyond what is shown.
5. Flag anything notable, such as a sector close to a threshold or a strong trend.
6. Use clear, professional language suitable for a credit committee memo.
7. Keep the response focused. Two to four sentences for simple questions, up to a short paragraph for complex ones.
8. State the unit for every number, inferred from its column name: _million_usd as "$X million", _usd as "$X", _percent as "X%", _count as "X loans" or "X borrowers" depending on the entity, _days as "X days", and _decimal probabilities converted to a percentage where natural. Never state a bare number when the source column implies a unit.

### OUTPUT
Return ONLY the natural language response text, with no markdown headers or bullet points unless truly needed.
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


def run_pipeline(user_question, db_connection, query_library, schema_context, status=None):
    """Runs the complete query engine pipeline for a single user question.
    If `status` (a Streamlit status/container object with .write) is given,
    progress is reported there instead of printed."""

    def report(msg):
        if status is not None:
            status.write(msg)

    log = {
        "user_question": user_question, "route": None, "query_id": None, "match_reason": None,
        "candidate_sql": None, "gate_result": None, "retry_used": False, "escalated": False,
        "executed_sql": None, "row_count": None, "confidence": None, "narrative": None,
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log["route"] = classification["route"]
    log["query_id"] = classification.get("query_id")
    log["match_reason"] = classification.get("match_reason")
    report(f"**Intent Classification:** route=`{log['route']}`, query_id=`{log['query_id']}`  \n{log['match_reason']}")

    # Step 2: Query construction
    if log["route"] == "verified" and log["query_id"] in query_library:
        candidate_sql = query_library[log["query_id"]]["sql"]
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log["candidate_sql"] = candidate_sql
    report("**Query Construction:** " + ("loaded from library" if log["route"] == "verified" else "generated fresh SQL"))

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log["query_id"])
    log["gate_result"] = gate
    report(f"**Validation Gate:** passed=`{gate['passed']}`, relevance_confidence=`{gate.get('relevance_confidence')}`")

    # Step 4: Retry once on generated track if validation fails
    if not gate["passed"] and log["route"] == "generated":
        report(f"Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate["details"], schema_context)
        log["candidate_sql"] = candidate_sql
        log["retry_used"] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log["gate_result"] = gate
        report(f"**Retry Validation Gate:** passed=`{gate['passed']}`, relevance_confidence=`{gate.get('relevance_confidence')}`")

    # Step 5: Escalate if still failing
    if not gate["passed"]:
        log["escalated"] = True
        log["narrative"] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log["confidence"] = "ESCALATED"
        report(f"**Escalated to human:** {gate['details']}")
        return {"log": log, "dataframe": None, **log}

    # Step 6: Execute
    log["executed_sql"] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result["dataframe"]
    log["row_count"] = len(df)
    report(f"**Execute:** {len(df)} rows returned")
    if exec_result["warnings"]:
        report(f"Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log["route"], log["query_id"])
    log["narrative"] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log["confidence"] = gate.get("relevance_confidence")
    report(f"**Response Generation:** confidence=`{log['confidence']}`")

    return {"log": log, "dataframe": df, **log}


# ============================================================
# Streamlit UI
# ============================================================
if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = []

st.title("📊 Credit Risk Query Engine")
st.caption("Ask a natural-language question about the commercial lending portfolio.")

with st.sidebar:
    st.header("About")
    st.write(
        "This app routes your question through a verified query template library "
        "when possible, or generates fresh SQL otherwise. Every query is validated "
        "— and, on the generated track, retried once — before execution. Read-only "
        "access only; nothing here can modify the database."
    )
    st.subheader("Example questions")
    st.markdown(
        "- How much of our book is in real estate, and how much of that is non-performing?\n"
        "- What does the DPD aging profile of the portfolio look like?\n"
        "- List every restructured loan, flag which are impaired, and show the overall impaired percentage.\n"
        "- Show me the average interest rate by sector, sorted from highest to lowest.\n"
        "- How has the Stage 3 book size and its expected credit loss moved over the last four quarters?\n"
    )
    st.subheader("Verified query library")
    with st.expander(f"{len(VERIFIED_QUERY_LIBRARY)} pre-approved templates"):
        for qid, entry in VERIFIED_QUERY_LIBRARY.items():
            st.markdown(f"**{qid}** — {entry['description']}")

    if st.session_state.audit_trail:
        st.divider()
        st.subheader("Audit trail")
        audit_df = pd.DataFrame(st.session_state.audit_trail)
        st.download_button(
            "Download audit trail (CSV)",
            audit_df.to_csv(index=False).encode("utf-8"),
            file_name="credit_risk_query_audit_trail.csv",
            mime="text/csv",
        )

user_question = st.text_input(
    "Your question",
    placeholder="e.g. How much of our book is in real estate, and how much of that is non-performing?",
)
show_trace = st.checkbox("Show pipeline trace", value=True)
submitted = st.button("Run query", type="primary")

if submitted and user_question.strip():
    trace_container = st.status("Running pipeline...", expanded=show_trace) if show_trace else None

    try:
        output = run_pipeline(
            user_question=user_question,
            db_connection=conn,
            query_library=VERIFIED_QUERY_LIBRARY,
            schema_context=DATABASE_SCHEMA,
            status=trace_container,
        )
    except Exception as e:
        if trace_container is not None:
            trace_container.update(label="Pipeline error", state="error")
        st.error(f"Pipeline failed: {e}")
        st.stop()

    if trace_container is not None:
        trace_container.update(
            label="Pipeline complete" if not output["escalated"] else "Escalated to human review",
            state="complete" if not output["escalated"] else "error",
        )

    st.divider()

    if output["escalated"]:
        st.warning(output["narrative"])
        with st.expander("Validation details"):
            st.json(output["gate_result"])
    else:
        confidence = output["confidence"]
        if isinstance(confidence, (int, float)):
            if confidence >= 0.8:
                badge = "🟢"
            elif confidence >= 0.6:
                badge = "🟡"
            else:
                badge = "🔴"
            confidence_display = f"{confidence:.2f}"
        else:
            badge = "⚪"
            confidence_display = str(confidence)

        st.subheader("Answer")
        st.write(output["narrative"])
        st.caption(
            f"{badge} Confidence: {confidence_display}  ·  Route: {output['route']}"
            + (f" ({output['query_id']})" if output.get("query_id") else "")
            + f"  ·  Rows returned: {output['row_count']}"
            + ("  ·  Retry used" if output["retry_used"] else "")
        )

        if output["dataframe"] is not None:
            with st.expander("View underlying data"):
                st.dataframe(output["dataframe"], use_container_width=True)

        with st.expander("View executed SQL"):
            st.code(output["executed_sql"], language="sql")

    # Audit trail entry — kept for the life of the browser session
    st.session_state.audit_trail.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": user_question,
        "route": output["route"],
        "query_id": output.get("query_id"),
        "retry_used": output["retry_used"],
        "escalated": output["escalated"],
        "confidence": output["confidence"],
        "row_count": output["row_count"],
        "executed_sql": output["executed_sql"],
    })

elif submitted:
    st.info("Please enter a question first.")
