import json
import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from google import genai
from dotenv import load_dotenv
import plotly.express as px

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY not set in .env")
    st.stop()

client = genai.Client(api_key=API_KEY)

# -------------------------------
# Google Sheet Configuration
# -------------------------------
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1PzYGEv2twgpmhvYUBTYh363WKExXzOqMyjVMtPA0uVo"
    "/export?format=csv&gid=1425471988"
)

# -------------------------------
# Security & Compliance Categories (Main → Sub)
# -------------------------------
MAIN_CATEGORIES = [
    "Security Dashboard",
    "Compliance",
    "Manage Vehicles / Drivers / Vendors",
]
SUB_BY_MAIN = {
    "Security Dashboard": [
        "Safe Reach Confirmation", "Employee Security Rules", "SOS Mobile Alert",
        "SOS Device Alert", "SOS Fixed Device Alert", "SOS Escalation Matrix",
        "Women Travelling Alone", "Vehicle Stoppage Alert", "Over Speeding Alert",
        "Unsafe Zone Alert", "Employee Geofence Violation",
        "Analytics Issue (Security)",
    ],
    "Compliance": [
        "Vehicle / Driver Configuration", "EHS", "Column Configuration",
        "Approval Dashboard", "Email Configuration", "Compliance UI Issue",
        "Wrong Compliance Status", "Compliance Summary Page", "Compliance Details Edit",
        "Audit / History Issue", "Analytics Issue (Compliance)",
    ],
    "Manage Vehicles / Drivers / Vendors": [
        "Vendor Page Issue", "Vehicle Creation Issue", "Vehicle Edit Issue",
        "Vehicle Activation Issue", "Vehicle Deactivation Issue", "Driver Activation Issue",
        "Driver Edit Issue", "Access Issue",
    ],
}
SUB_TO_MAIN = {sub: main for main, subs in SUB_BY_MAIN.items() for sub in subs}
AVAILABLE_CATEGORIES = [s for subs in SUB_BY_MAIN.values() for s in subs]
PRIORITIES = ["Critical", "High", "Medium", "Low"]
REQUIRED_COLS = ["Key", "Summary", "Description", "Status", "Created"]
OPTIONAL_COLS = ["Priority", "Assignee", "Resolved", "Resolution Type", "Resolution type"]
RESOLUTION_TYPE_COLS = ["Resolution Type", "Resolution type"]
DEFAULT_MAIN = "Manage Vehicles / Drivers / Vendors"
STATUS_OPEN_PATTERN = re.compile(r"open|in progress", re.I)
STATUS_RESOLVED_PATTERN = re.compile(r"resolved|closed|done|completed")
_CATEGORIES_PROMPT_FRAGMENT = "\n".join(f"- {c}" for c in AVAILABLE_CATEGORIES)

# Keyword-based category mapping (no AI tokens used)
CATEGORY_KEYWORDS = [
    # 🔐 Security Dashboard
    (["safe reach", "src", "arrival confirmation"], "Safe Reach Confirmation"),
    (["security rule", "escort rule", "night shift rule"], "Employee Security Rules"),
    (["sos mobile", "panic app", "mobile sos"], "SOS Mobile Alert"),
    (["sos device", "panic device", "hardware sos"], "SOS Device Alert"),
    (["fixed sos", "vehicle sos button", "cabin sos"], "SOS Fixed Device Alert"),
    (["escalation matrix", "emergency escalation", "escalation flow", "ivr issue", "voice alert", "automated call"], "SOS Escalation Matrix"),
    (["women alone", "female escort", "lone female"], "Women Travelling Alone"),
    (["vehicle stopped", "long halt", "idle alert"], "Vehicle Stoppage Alert"),
    (["over speed", "speed breach", "rash driving"], "Over Speeding Alert"),
    (["unsafe zone", "blacklist area", "danger zone"], "Unsafe Zone Alert"),
    (["geofence breach", "location violation", "boundary breach"], "Employee Geofence Violation"),
    (["security dashboard report", "security analytics", "alert report"], "Analytics Issue (Security)"),

    # 📋 Compliance
    (["compliance config", "vehicle document", "driver document expiry"], "Vehicle / Driver Configuration"),
    (["ehs", "environment health safety"], "EHS"),
    (["column missing", "field configuration", "table column issue"], "Column Configuration"),
    (["approval pending", "approval flow", "compliance approval"], "Approval Dashboard"),
    (["compliance email", "notification email", "smtp issue"], "Email Configuration"),
    (["compliance page error", "ui misalignment", "screen issue"], "Compliance UI Issue"),
    (["incorrect status", "wrong compliance status", "status mismatch"], "Wrong Compliance Status"),
    (["compliance summary", "overview page issue"], "Compliance Summary Page"),
    (["edit compliance", "update compliance record"], "Compliance Details Edit"),
    (["audit trail missing", "history log issue"], "Audit / History Issue"),
    (["compliance report error", "compliance analytics"], "Analytics Issue (Compliance)"),

    # 🚗 Manage Vehicles / Drivers / Vendors
    (["vendor page error", "vendor profile issue"], "Vendor Page Issue"),
    (["vehicle not created", "add vehicle error"], "Vehicle Creation Issue"),
    (["vehicle update failed", "edit vehicle issue"], "Vehicle Edit Issue"),
    (["activate vehicle error", "vehicle status not active"], "Vehicle Activation Issue"),
    (["deactivate vehicle", "vehicle inactive issue"], "Vehicle Deactivation Issue"),
    (["driver activate error", "driver status issue"], "Driver Activation Issue"),
    (["driver profile edit", "update driver info"], "Driver Edit Issue"),
    (["permission denied", "access error", "role issue", "login restriction"], "Access Issue"),
]


def categorize_ticket(summary: str, description: str) -> str:
    """
    Keyword-based ticket categorization.
    Returns one of the known subcategories when a keyword matches.
    If no keyword matches, uses Gemini to categorize into one of the existing subcategories.
    """
    text = f"{summary or ''} {description or ''}".lower()
    for keywords, category in CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in text:
                return category
    # No keyword match — use Gemini as a backup to choose closest category
    return classify_ticket_via_gemini(summary, description)


def _parse_json_from_response(text: str):
    """Strip markdown code fences and parse JSON."""
    text = (text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1]
            if text.lower().startswith("json"):
                text = text[4:]
    return json.loads(text.strip())


def classify_ticket_via_gemini(summary: str, description: str) -> str | None:
    """
    Use Gemini to pick the closest category name from AVAILABLE_CATEGORIES.
    Only called when keyword-based classification returns no match.
    """
    prompt = f"""You are a Security & Compliance ticket classification engine.
Your job is to categorize the ticket into EXACTLY ONE category from the predefined list.

AVAILABLE CATEGORIES (choose exactly one, use the exact name):
{_CATEGORIES_PROMPT_FRAGMENT}

Reply with ONLY a valid JSON object, no other text:
{{"category": "<exact category name from list>"}}

Ticket Summary: {summary}
Ticket Description: {description or 'N/A'}"""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        out = _parse_json_from_response(response.text)
        cat = (out.get("category") or "").strip()
        if cat in AVAILABLE_CATEGORIES:
            return cat
    except Exception:
        pass
    return None


# Resolution type categorization
RESOLUTION_TYPE_MAP = {
    "tech fix": "Tech Fix",
    "techfix": "Tech Fix",
    "technical fix": "Tech Fix",
    "backend config": "Enablement",
    "backend configuration": "Enablement",
    "enablement": "Enablement",
    "configuration": "Enablement",
    "config": "Enablement",
}


def categorize_resolution_type(res_type):
    """Map resolution type to Tech Fix, Enablement, or Non Tech Fix."""
    if not res_type or pd.isna(res_type):
        return "Non Tech Fix"
    res_lower = str(res_type).strip().lower()
    for key, mapped in RESOLUTION_TYPE_MAP.items():
        if key in res_lower:
            return mapped
    return "Non Tech Fix"


# -------------------------------
# Email Functionality
# -------------------------------
def generate_email_html(analytics):
    """Generate HTML email with analytics. Order: 1) Main vs resolution type, 2) Top 10 non-tech last week, 3) Subcategory comparison last 2 weeks, 4) Actions to be Taken (3–4 crisp bullets)."""
    dr = analytics.get("analysis_date_range") or {}
    start_d, end_d = dr.get("start", ""), dr.get("end", "")
    date_range_str = f" ({start_d} to {end_d})" if start_d and end_d else ""

    res_by_main = analytics.get("resolution_type_by_main_category", {})
    top10_non_tech = analytics.get("last_week_top_non_tech_subcategories", [])
    sub_comp = analytics.get("subcategory_comparison_last_2_weeks", [])
    actions_crisp = analytics.get("actions_crisp", [])
    top3_summaries = analytics.get("top3_subcategory_summaries", [])

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
            .container { max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #1e3a5f; border-bottom: 3px solid #3b82f6; padding-bottom: 10px; }
            h2 { color: #475569; margin-top: 30px; }
            table { width: 100%; border-collapse: collapse; margin: 15px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f8fafc; font-weight: bold; color: #1e3a5f; }
            tr:hover { background-color: #f1f5f9; }
            .action-item { margin: 10px 0; padding: 10px; background-color: #fff7ed; border-left: 4px solid #f59e0b; }
            .issue-sample { font-size: 0.9rem; color: #475569; margin: 4px 0 0 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ Safety Ticket Analytics Report</h1>
            <p><strong>Date range:</strong>""" + (date_range_str or " (selected in UI)") + """</p>
            <p><strong>Generated:</strong> """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    """

    # 1st matrix: Main category vs resolution type — sorted by decreasing overall count
    if res_by_main:
        main_totals = {m: sum(t.get(rt, 0) for rt in ["Tech Fix", "Enablement", "Non Tech Fix"]) for m, t in res_by_main.items()}
        res_by_main_sorted = sorted(res_by_main.items(), key=lambda x: -main_totals.get(x[0], 0))
        html += """
            <h2>1. Main Category vs Resolution Type</h2>
            <table>
                <tr><th>Main Category</th><th>Tech Fix</th><th>Enablement</th><th>Non Tech Fix</th></tr>
        """
        for main, types in res_by_main_sorted:
            tech = types.get("Tech Fix", 0)
            enable = types.get("Enablement", 0)
            non_tech = types.get("Non Tech Fix", 0)
            html += f"<tr><td>{main}</td><td>{tech}</td><td>{enable}</td><td>{non_tech}</td></tr>"
        html += "</table>"

    # 2nd matrix: Top 10 non-tech subcategories last week — already by count desc
    if top10_non_tech:
        html += """
            <h2>2. Top 10 Non-Tech Categories (Last Week)</h2>
            <table>
                <tr><th>Subcategory</th><th>Count</th></tr>
        """
        for cat, count in top10_non_tech:
            html += f"<tr><td>{cat}</td><td>{count}</td></tr>"
        html += "</table>"

    # 3rd matrix: Subcategory comparison last 2 weeks — sorted by decreasing overall count (week1+week2)
    if sub_comp:
        sub_comp_sorted = sorted(sub_comp, key=lambda x: -(x.get("week1", 0) + x.get("week2", 0)))
        html += """
            <h2>3. Subcategory Comparison (Last 2 Weeks)</h2>
            <table>
                <tr><th>Subcategory</th><th>Week 1 (Previous)</th><th>Week 2 (Last Week)</th><th>Change</th></tr>
        """
        for row in sub_comp_sorted:
            sub = row.get("subcategory", "")
            w1 = row.get("week1", 0)
            w2 = row.get("week2", 0)
            ch = row.get("change", w2 - w1)
            ch_str = f"+{ch}" if ch > 0 else str(ch)
            html += f"<tr><td>{sub}</td><td>{w1}</td><td>{w2}</td><td>{ch_str}</td></tr>"
        html += "</table>"

    # 4. Actions to be Taken — crisp bullets + ticket summary of TOP 3 subcategories
    html += """
            <h2>4. Actions to be Taken</h2>
    """
    if actions_crisp:
        html += """
            <ul style="margin: 12px 0; padding-left: 24px;">
        """
        for bullet in actions_crisp:
            html += f"<li class=\"action-item\" style=\"margin: 8px 0;\">{bullet}</li>"
        html += "</ul>"
    if top3_summaries:
        html += """
            <p style="margin-top: 16px; font-weight: 600;">Ticket summary of TOP 3 subcategories:</p>
            <ul style="margin: 8px 0; padding-left: 24px;">
        """
        for item in top3_summaries:
            sub = item.get("subcategory", "")
            summary = item.get("summary", "") or "—"
            html += f"<li style=\"margin: 6px 0;\"><strong>{sub}:</strong> {summary}</li>"
        html += "</ul>"

    html += """
        </div>
    </body>
    </html>
    """
    return html


def _strip_invisible_chars(s: str) -> str:
    """
    Remove invisible Unicode chars (like zero-width space) that can
    break email header/body encoding.
    """
    if s is None:
        return ""
    return re.sub(r"[\u200B-\u200D\uFEFF]", "", str(s)).strip()


def _ai_summarize_issues(texts: list) -> str:
    """Use Gemini to summarize the issue raised by the client in one line. Prefer description content; do not just repeat subject/summary."""
    if not texts:
        return "Review tickets in this category for details."
    parts = [str(t)[:300].strip() for t in texts if t and str(t).strip()][:15]
    combined = "\n".join(parts)
    if not combined.strip():
        return "Review tickets in this category for details."
    prompt = f"""Read the following ticket content (descriptions or details). Summarize in ONE line what issue the client raised or what action is needed. Be specific and useful. Do not just repeat the subject/title—focus on the actual issue or request.

Ticket content:
{combined}

One-line summary of issue raised / action needed:"""
    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        out = (response.text or "").strip()
        if out:
            return out[:400]
        return "Various client-reported issues; review tickets for details."
    except Exception:
        return "Various client-reported issues; review tickets for details."


def _get_ticket_text_columns(df):
    """Return (description_col, summary_col) - description preferred for summarization."""
    # Try common column names (case-insensitive)
    cols_lower = {c.strip().lower(): c for c in df.columns if isinstance(c, str)}
    desc_col = cols_lower.get("description") or cols_lower.get("synopsis") or cols_lower.get("details")
    if desc_col and desc_col not in df.columns:
        desc_col = None
    elif not desc_col and "Description" in df.columns:
        desc_col = "Description"
    summary_col = cols_lower.get("summary") or ("Summary" if "Summary" in df.columns else None)
    return (desc_col, summary_col)


def get_top3_subcategory_summaries(df, analytics):
    """Return list of {subcategory, summary} for top 3 subcategories by non-tech count (for email Actions section)."""
    items = list(analytics.get("email_action_items") or [])[:3]
    if not items:
        return []
    res_type_col = next((c for c in RESOLUTION_TYPE_COLS if c in df.columns), None)
    if not res_type_col:
        return [{"subcategory": it.get("subcategory", ""), "summary": ""} for it in items]
    df = df.copy()
    df["_res_type_cat"] = df[res_type_col].apply(categorize_resolution_type)
    non_tech_df = df[df["_res_type_cat"] == "Non Tech Fix"]
    desc_col, summary_col = _get_ticket_text_columns(df)
    if not desc_col and not summary_col:
        return [{"subcategory": it.get("subcategory", ""), "summary": ""} for it in items]
    result = []
    for it in items:
        sub = it.get("subcategory", "")
        subset = non_tech_df[non_tech_df["AI Category"] == sub]
        texts = []
        for _, row in subset.head(12).iterrows():
            d = (desc_col and row.get(desc_col)) or ""
            s = (summary_col and row.get(summary_col)) or ""
            if pd.notna(d) and str(d).strip():
                texts.append(str(d).strip())
            elif pd.notna(s) and str(s).strip():
                texts.append(str(s).strip())
        result.append({"subcategory": sub, "summary": _ai_summarize_issues(texts)})
    return result


def enrich_email_action_items_with_ai_summaries(df, analytics):
    """Add ai_summary to each email_action_items entry using Gemini. Returns updated list."""
    items = list(analytics.get("email_action_items") or [])
    if not items or "_res_type_cat" not in df.columns:
        return items
    res_type_col = next((c for c in RESOLUTION_TYPE_COLS if c in df.columns), None)
    if not res_type_col:
        return items
    df = df.copy()
    df["_res_type_cat"] = df[res_type_col].apply(categorize_resolution_type)
    non_tech_df = df[df["_res_type_cat"] == "Non Tech Fix"]
    if non_tech_df.empty:
        return items
    desc_col, summary_col = _get_ticket_text_columns(df)
    # Need at least one column to get ticket content
    if not desc_col and not summary_col:
        return [
            {**item, "ai_summary": "Add Description or Summary column in sheet for insights."}
            for item in items
        ]
    result = []
    for item in items:
        sub = item.get("subcategory", "")
        subset = non_tech_df[non_tech_df["AI Category"] == sub]
        texts = []
        for _, row in subset.head(12).iterrows():
            # Prefer description; fall back to summary so we always have something to summarize
            d = (desc_col and row.get(desc_col)) or ""
            s = (summary_col and row.get(summary_col)) or ""
            if pd.notna(d) and str(d).strip():
                texts.append(str(d).strip())
            elif pd.notna(s) and str(s).strip():
                texts.append(str(s).strip())
        ai_summary = _ai_summarize_issues(texts)
        result.append({
            "subcategory": sub,
            "non_tech_count": item.get("non_tech_count", 0),
            "ai_summary": ai_summary,
        })
    return result


def send_email(recipient_email, analytics, smtp_server=None, smtp_port=None, smtp_user=None, smtp_password=None, sender_email=None):
    """Send analytics email (UTF-8 safe, strips zero-width chars)."""
    smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
    smtp_user = _strip_invisible_chars(smtp_user or os.getenv("SMTP_USER"))
    smtp_password = _strip_invisible_chars(smtp_password or os.getenv("SMTP_PASSWORD"))
    sender_email = _strip_invisible_chars(sender_email or os.getenv("SENDER_EMAIL", smtp_user))
    recipient_email = _strip_invisible_chars(recipient_email)
    
    if not smtp_user or not smtp_password:
        return False, "SMTP credentials not configured. Set SMTP_USER, SMTP_PASSWORD in .env"
    
    try:
        msg = MIMEMultipart("alternative")
        dr = (analytics or {}).get("analysis_date_range") or {}
        start_d, end_d = dr.get("start"), dr.get("end")
        if start_d and end_d:
            subject = _strip_invisible_chars(
                f"Safety Analytics Report ({start_d} to {end_d})"
            )
        else:
            subject = _strip_invisible_chars(
                f"Safety Analytics Report - {datetime.now().strftime('%Y-%m-%d')}"
            )
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email
        
        html_content = _strip_invisible_chars(generate_email_html(analytics))
        # Force UTF-8 so emojis/special chars don't trigger ASCII encoding errors
        html_part = MIMEText(html_content, "html", "utf-8")
        msg.attach(html_part)
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        return True, "Email sent successfully!"
    except Exception as e:
        return False, f"Error sending email: {str(e)}"

# -------------------------------
# Google Sheet Fetch Function
# -------------------------------
def fetch_issues_from_google_sheet(sheet_csv_url, max_results=None):
    """Load sheet; ensure required + optional columns exist; return selected columns."""
    try:
        df = pd.read_csv(sheet_csv_url)
    except Exception as e:
        st.error(f"Error reading Google Sheet CSV: {e}")
        st.stop()
    for col in REQUIRED_COLS + OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = ""
    use_cols = [c for c in REQUIRED_COLS + OPTIONAL_COLS if c in df.columns]
    df = df[use_cols]
    if max_results is not None:
        df = df.head(max_results)
    return df


# -------------------------------
# Analytics from Classified Tickets (vectorized)
# -------------------------------
def compute_analytics(df, analysis_start=None, analysis_end=None):
    """
    df must have: AI Category, AI Priority, Status, Created, Assignee.
    Optional: Resolved, Resolution Type / Resolution type.
    analysis_start, analysis_end: date range selected in UI (for email subject and report).
    """
    df = df.copy()
    n = len(df)
    now = pd.Timestamp(datetime.now())
    cat = df["AI Category"].fillna("Unknown").astype(str)
    prio = df["AI Priority"].fillna("Medium").astype(str)
    status_raw = df["Status"].fillna("").astype(str).str.strip()
    status_lower = status_raw.str.lower()
    is_open = status_lower.str.contains(STATUS_OPEN_PATTERN, regex=True, na=False) | (status_raw == "")
    is_resolved = status_lower.str.contains(STATUS_RESOLVED_PATTERN, regex=True, na=False)
    assignee = df["Assignee"].fillna("Unassigned").astype(str).str.strip().replace("", "Unassigned")

    df["_main"] = cat.map(SUB_TO_MAIN).fillna(DEFAULT_MAIN)
    created_dt = pd.to_datetime(df["Created"], errors="coerce")
    resolved_dt = pd.to_datetime(df["Resolved"], errors="coerce") if "Resolved" in df.columns else pd.NaT
    age_days = (now - created_dt).dt.days.fillna(0)

    category_distribution = {k: int(v) for k, v in cat.value_counts().to_dict().items()}
    priority_distribution = {k: int(v) for k, v in prio.value_counts().to_dict().items()}
    risk_heatmap = df.groupby(["AI Category", "AI Priority"], dropna=False).size().unstack(fill_value=0)
    risk_heatmap = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in risk_heatmap.to_dict("index").items()}

    main_sub = df.groupby(["_main", "AI Category"], dropna=False).size()
    main_counts = defaultdict(dict)
    for (main, sub), count in main_sub.items():
        main_counts[main][sub] = int(count)
    main_counts = {k: dict(v) for k, v in main_counts.items()}

    assignee_df = pd.DataFrame({"assignee": assignee, "prio": prio, "resolved": is_resolved})
    assignee_df["_hc"] = assignee_df["prio"].isin(["Critical", "High"])
    assignee_totals = assignee_df.groupby("assignee").agg(
        total=("assignee", "count"),
        high_critical=("_hc", "sum"),
        resolved=("resolved", "sum"),
    ).to_dict("index")
    team_load = {a: {"total": int(v["total"]), "high_critical": int(v["high_critical"]), "resolved": int(v["resolved"])} for a, v in assignee_totals.items()}

    open_count = int(is_open.sum())
    critical_open = int(((prio == "Critical") & is_open).sum())
    high_over_3_days = int(((prio == "High") & (age_days > 3)).sum())
    total_sla_risk = critical_open + high_over_3_days

    weekly_by_main = defaultdict(lambda: defaultdict(int))
    valid_created = created_dt.notna()
    if valid_created.any():
        temp = df.loc[valid_created].copy()
        temp["_week"] = created_dt.loc[valid_created].dt.strftime("%Y-W%W")
        for (week_key, main), count in temp.groupby(["_week", "_main"]).size().items():
            weekly_by_main[week_key][main] += int(count)

    weekly_resolution_by_assignee = defaultdict(lambda: defaultdict(int))
    if "Resolved" in df.columns and resolved_dt.notna().any() and is_resolved.any():
        use = resolved_dt.notna() & is_resolved
        if use.any():
            temp = df.loc[use].copy()
            temp["_week"] = resolved_dt.loc[use].dt.strftime("%Y-W%W")
            temp["_assignee"] = assignee.loc[use]
            for (week_key, a), count in temp.groupby(["_week", "_assignee"]).size().items():
                weekly_resolution_by_assignee[week_key][a] += int(count)

    res_type_col = next((c for c in RESOLUTION_TYPE_COLS if c in df.columns and df[c].notna().any()), None)
    if res_type_col:
        resolution_type_distribution = {k: int(v) for k, v in df[res_type_col].replace("", pd.NA).dropna().astype(str).str.strip().value_counts().to_dict().items()}
        df["_res_type_cat"] = df[res_type_col].apply(categorize_resolution_type)
        res_type_cat_dist = df["_res_type_cat"].value_counts().to_dict()
        res_type_cat_dist = {k: int(v) for k, v in res_type_cat_dist.items()}
        res_type_by_category = df.groupby(["AI Category", "_res_type_cat"], dropna=False).size().unstack(fill_value=0)
        res_type_by_category = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in res_type_by_category.to_dict("index").items()}
        # Sort by overall count (highest to lowest) for Resolution Type vs Category
        cat_totals = {c: sum(t.get(rt, 0) for rt in ["Tech Fix", "Enablement", "Non Tech Fix"]) for c, t in res_type_by_category.items()}
        res_type_by_category = dict(sorted(res_type_by_category.items(), key=lambda x: -cat_totals.get(x[0], 0)))
        # Main category vs resolution type (for email matrix 1)
        res_type_by_main = df.groupby(["_main", "_res_type_cat"], dropna=False).size().unstack(fill_value=0)
        resolution_type_by_main_category = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in res_type_by_main.to_dict("index").items()}
    else:
        resolution_type_distribution = {}
        res_type_cat_dist = {}
        res_type_by_category = {}
        resolution_type_by_main_category = {}

    top_areas = sorted(category_distribution.items(), key=lambda x: -x[1])[:10]

    # Last week top issues and 2-week comparison (calendar weeks: Monday–Sunday)
    last_week_top = []
    last_2_weeks_comparison = {}
    last_week_top_non_tech = []
    subcategory_comparison_last_2_weeks = []
    if valid_created.any():
        # Use calendar weeks with Monday as the first day and Sunday as the last
        today = now.normalize()
        this_monday = today - timedelta(days=today.weekday())  # current week Monday
        last_monday = this_monday - timedelta(days=7)          # last week Monday
        prev_monday = last_monday - timedelta(days=7)          # previous week Monday
        last_sunday = this_monday - timedelta(days=1)          # last week Sunday
        prev_sunday = last_monday - timedelta(days=1)          # previous week Sunday
        
        # Last week (Monday–Sunday) top issues
        last_week_mask = (created_dt >= last_monday) & (created_dt <= last_sunday)
        if last_week_mask.any():
            last_week_df = df.loc[last_week_mask]
            last_week_cats = last_week_df["AI Category"].value_counts()
            last_week_top = [(cat, int(count)) for cat, count in last_week_cats.head(10).items()]
        
        # Last 2 full calendar weeks comparison (Week1 = previous week, Week2 = last week)
        week1_mask = (created_dt >= prev_monday) & (created_dt <= prev_sunday)
        week2_mask = (created_dt >= last_monday) & (created_dt <= last_sunday)
        if week1_mask.any() or week2_mask.any():
            week1_cats = df.loc[week1_mask, "_main"].value_counts().to_dict() if week1_mask.any() else {}
            week2_cats = df.loc[week2_mask, "_main"].value_counts().to_dict() if week2_mask.any() else {}
            all_mains = set(week1_cats.keys()) | set(week2_cats.keys())
            for main in all_mains:
                last_2_weeks_comparison[main] = {
                    "week1": int(week1_cats.get(main, 0)),
                    "week2": int(week2_cats.get(main, 0)),
                }

        # Top 10 non-tech subcategories last week (calendar week)
        last_week_top_non_tech = []
        if "_res_type_cat" in df.columns and last_week_mask.any():
            last_week_df = df.loc[last_week_mask]
            non_tech_last_week = last_week_df[last_week_df["_res_type_cat"] == "Non Tech Fix"]
            if not non_tech_last_week.empty:
                sub_counts = non_tech_last_week["AI Category"].value_counts()
                last_week_top_non_tech = [(str(cat), int(cnt)) for cat, cnt in sub_counts.head(10).items()]

        # Subcategory comparison last 2 weeks (subcategory = AI Category)
        subcategory_comparison_last_2_weeks = []
        if week1_mask.any() or week2_mask.any():
            w1_subs = df.loc[week1_mask, "AI Category"].value_counts().to_dict() if week1_mask.any() else {}
            w2_subs = df.loc[week2_mask, "AI Category"].value_counts().to_dict() if week2_mask.any() else {}
            all_subs = set(w1_subs.keys()) | set(w2_subs.keys())
            for sub in all_subs:
                s1, s2 = int(w1_subs.get(sub, 0)), int(w2_subs.get(sub, 0))
                subcategory_comparison_last_2_weeks.append({
                    "subcategory": str(sub),
                    "week1": s1,
                    "week2": s2,
                    "change": s2 - s1,
                })
            subcategory_comparison_last_2_weeks.sort(key=lambda x: (-x["week2"], -x["change"]))

    # Email action items: top 5 subcategories by non-tech count + sample issues raised by client
    email_action_items = []
    if "_res_type_cat" in df.columns and "Summary" in df.columns:
        non_tech_df = df[df["_res_type_cat"] == "Non Tech Fix"]
        if not non_tech_df.empty:
            sub_counts = non_tech_df["AI Category"].value_counts()
            top5_subs = list(sub_counts.head(5).index)
            for sub in top5_subs:
                subset = non_tech_df[non_tech_df["AI Category"] == sub]
                samples = []
                for _, row in subset.head(5).iterrows():
                    s = (row.get("Summary") or row.get("Description") or "")
                    if isinstance(s, str) and s.strip():
                        samples.append(s.strip()[:120] + ("..." if len(s.strip()) > 120 else ""))
                email_action_items.append({
                    "subcategory": str(sub),
                    "non_tech_count": int(sub_counts[sub]),
                    "sample_issues": samples[:5],
                })

    actions = []
    if critical_open > 0:
        actions.append(f"Urgent: {critical_open} Critical ticket(s) still open — assign and resolve immediately.")
    if high_over_3_days > 0:
        actions.append(f"High priority: {high_over_3_days} High-priority ticket(s) older than 3 days — review and escalate.")
    if open_count > n * 0.6:
        actions.append(f"Backlog: {open_count} open tickets ({100*open_count/n:.0f}% of total) — consider reassignment or focus sprint.")
    overloaded = [a for a, v in team_load.items() if v["high_critical"] >= 3 and a != "Unassigned"]
    if overloaded:
        actions.append(f"Team load: {', '.join(overloaded)} have 3+ Critical/High tickets — rebalance or add support.")
    if not actions:
        actions.append("No critical actions. Continue monitoring SLA and backlog.")

    # Crisp actions for email (3–4 bullets from matrices): increased this week, focus to reduce, new categories
    actions_crisp = []
    # 1) Subcategories that increased this week → focus on solving/reducing these
    increased = [r for r in subcategory_comparison_last_2_weeks if r.get("change", 0) > 0]
    increased.sort(key=lambda x: -x.get("change", 0))
    if increased:
        top_increased = [r["subcategory"] for r in increased[:5]]
        actions_crisp.append(f"Subcategories with increased ticket count this week: {', '.join(top_increased)}.")
    # 2) New categories of tickets this week (week2 > 0, week1 == 0)
    new_this_week = [r["subcategory"] for r in subcategory_comparison_last_2_weeks if r.get("week1", 0) == 0 and r.get("week2", 0) > 0]
    if new_this_week:
        actions_crisp.append(f"New ticket categories this week: {', '.join(new_this_week[:8])}.")
    # 3) High-volume non-tech subcategory (from last week)
    if last_week_top_non_tech and len(actions_crisp) < 4:
        top_sub = last_week_top_non_tech[0][0]
        top_cnt = last_week_top_non_tech[0][1]
        actions_crisp.append(f"Highest non-tech volume last week: {top_sub} ({top_cnt}).")
    # Cap at 4 bullets; ensure at least one if we have any data
    if not actions_crisp and (subcategory_comparison_last_2_weeks or resolution_type_by_main_category):
        actions_crisp.append("Review matrices above for subcategories with rising counts or high non-tech volume.")
    actions_crisp = actions_crisp[:4]

    # analysis_date_range for email subject and report
    analysis_date_range = {}
    if analysis_start is not None and analysis_end is not None:
        analysis_date_range = {"start": str(analysis_start), "end": str(analysis_end)}

    return {
        "category_distribution": category_distribution,
        "priority_distribution": priority_distribution,
        "risk_heatmap": risk_heatmap,
        "sla_summary": {"total_sla_risk": total_sla_risk, "critical_open": critical_open, "high_over_3_days": high_over_3_days},
        "team_load": team_load,
        "main_sub_counts": main_counts,
        "weekly_category": dict(weekly_by_main),
        "weekly_assignee_resolution": dict(weekly_resolution_by_assignee),
        "open_tickets_count": open_count,
        "top_contributing_areas": top_areas,
        "actions": actions,
        "resolution_type_distribution": resolution_type_distribution,
        "resolution_type_categorized": res_type_cat_dist,
        "resolution_type_by_category": res_type_by_category,
        "resolution_type_by_main_category": resolution_type_by_main_category if res_type_col else {},
        "last_week_top_issues": last_week_top,
        "last_2_weeks_comparison": last_2_weeks_comparison,
        "last_week_top_non_tech_subcategories": last_week_top_non_tech if valid_created.any() else [],
        "subcategory_comparison_last_2_weeks": subcategory_comparison_last_2_weeks if valid_created.any() else [],
        "email_action_items": email_action_items,
        "actions_crisp": actions_crisp,
        "analysis_date_range": analysis_date_range,
    }


def send_last_two_weeks_email_default_recipient():
    """
    Fetch issues for the last 2 full weeks (Monday–Sunday) from today,
    run classification + analytics, and send the email to the default recipient.
    """
    df_raw = fetch_issues_from_google_sheet(SHEET_CSV_URL)
    if df_raw.empty:
        return False, "No issues found in the sheet."

    created_dt = pd.to_datetime(df_raw["Created"], errors="coerce")
    valid = created_dt.notna()
    if not valid.any():
        return False, "No valid Created dates in the sheet."

    today = datetime.now().date()
    this_monday = today - timedelta(days=today.weekday())
    last_monday = this_monday - timedelta(days=7)
    prev_monday = last_monday - timedelta(days=7)
    last_sunday = this_monday - timedelta(days=1)

    start_date = prev_monday
    end_date = last_sunday

    mask = valid & (created_dt.dt.date >= start_date) & (created_dt.dt.date <= end_date)
    df = df_raw.loc[mask].copy()
    if df.empty:
        return False, f"No tickets with Created date between {start_date} and {end_date}."

    categories = []
    for _, row in df.iterrows():
        cat = categorize_ticket(row["Summary"], row["Description"])
        categories.append(cat)
    df["AI Category"] = categories

    if "Priority" in df.columns:
        df["AI Priority"] = (
            df["Priority"].fillna("Medium").astype(str).str.strip().replace("", "Medium")
        )
    else:
        df["AI Priority"] = "Medium"

    df["Main Category"] = df["AI Category"].map(SUB_TO_MAIN).fillna(DEFAULT_MAIN)
    if "Priority" in df.columns:
        if (df["Priority"].astype(str).str.strip() == "").all():
            df["Priority"] = df["AI Priority"]
        else:
            df["Priority"] = df["Priority"].fillna(df["AI Priority"])
    else:
        df["Priority"] = df["AI Priority"]

    assignee_raw = df["Assignee"].fillna("").astype(str).str.strip()
    df["Assignee"] = assignee_raw.replace("", "Unassigned")

    analytics = compute_analytics(df, analysis_start=start_date, analysis_end=end_date)
    analytics["analysis_date_range"] = {
        "start": str(start_date),
        "end": str(end_date),
    }

    recipient = "sukeerthi.reddy@moveinsync.com"
    return send_email(recipient, analytics)


# If this script is run directly (not only via Streamlit), send the email once.
if __name__ == "__main__":
    success, message = send_last_two_weeks_email_default_recipient()
    print(message)

# -------------------------------
# Dashboard CSS
# -------------------------------
st.set_page_config(page_title="Security & Compliance Dashboard", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    .kpi-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        margin-bottom: 0.5rem;
    }
    .kpi-card h3 { color: #94a3b8; font-size: 0.85rem; font-weight: 600; margin: 0 0 0.25rem 0; }
    .kpi-card .value { color: #f1f5f9; font-size: 1.75rem; font-weight: 700; }
    .main-cat-section {
        background: #0f172a;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        border: 1px solid #334155;
    }
    .main-cat-section h4 { color: #38bdf8; margin: 0 0 0.5rem 0; font-size: 1.05rem; }
    .sub-cat-list { color: #cbd5e1; font-size: 0.9rem; margin: 0.25rem 0 0 1rem; }
    .action-item {
        background: #1e293b;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #f59e0b;
        color: #e2e8f0;
    }
    .stMetric { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid #cbd5e1; color: #0f172a; }
    .stMetric label { color: #475569 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #0f172a !important; }
    .stMetric [data-testid="stMetricDelta"] { color: #64748b !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🛡️ Safety Ticket Dashboard")
st.caption("Classify tickets from Google Sheet · Main categories, weekly trends, and actionable insights.")

# Default date range (no sheet fetch until user runs analysis)
_today = datetime.now().date()
_default_start = _today - timedelta(days=90)
_default_end = _today

st.markdown("### 📅 Select date range")
col_start, col_end, _ = st.columns([1, 1, 2])
with col_start:
    start_date = st.date_input("From", value=_default_start, key="start_date")
with col_end:
    end_date = st.date_input("To", value=_default_end, key="end_date")
if start_date > end_date:
    st.warning("From date must be before or equal to To date.")
    start_date, end_date = _default_start, _default_end

run_clicked = st.button("Run analysis")
if run_clicked:
    with st.spinner("Fetching data and running analysis..."):
        df_raw = fetch_issues_from_google_sheet(SHEET_CSV_URL)
        if df_raw.empty:
            st.error("No issues found in the sheet.")
        else:
            created_dt = pd.to_datetime(df_raw["Created"], errors="coerce")
            valid_dates = created_dt.notna()
            if valid_dates.any():
                in_range = (created_dt.dt.date >= start_date) & (created_dt.dt.date <= end_date)
                df = df_raw.loc[valid_dates & in_range].copy()
            else:
                df = df_raw.copy()
            if df.empty:
                st.warning(f"No tickets with **Created** date between **{start_date}** and **{end_date}**. Adjust the date range.")
            else:
                categories = []
                for _, row in df.iterrows():
                    cat = categorize_ticket(row["Summary"], row["Description"])
                    categories.append(cat)
                df["AI Category"] = categories
                if "Priority" in df.columns:
                    df["AI Priority"] = (
                        df["Priority"].fillna("Medium").astype(str).str.strip().replace("", "Medium")
                    )
                else:
                    df["AI Priority"] = "Medium"
                df["Main Category"] = df["AI Category"].map(SUB_TO_MAIN).fillna(DEFAULT_MAIN)
                df["Priority"] = df["AI Priority"] if (df["Priority"].astype(str).str.strip() == "").all() else df["Priority"].fillna(df["AI Priority"])
                assignee_raw = df["Assignee"].fillna("").astype(str).str.strip()
                df["Assignee"] = assignee_raw.replace("", "Unassigned")
                analytics = compute_analytics(df, analysis_start=start_date, analysis_end=end_date)
                if "analysis_result" not in st.session_state:
                    st.session_state["analysis_result"] = {}
                st.session_state["analysis_result"] = {"df": df, "analytics": analytics, "start_date": start_date, "end_date": end_date}
                st.rerun()

if not st.session_state.get("analysis_result"):
    st.info("Select date range above and click **Run analysis** to load data.")
else:
    result = st.session_state["analysis_result"]
    df = result["df"]
    analytics = result["analytics"]
    start_date = result["start_date"]
    end_date = result["end_date"]
    st.success(f"Showing **{len(df)}** issues (Created between **{start_date}** and **{end_date}**).")
    main_counts = analytics["main_sub_counts"]
    weekly_cat = analytics["weekly_category"]
    weekly_assignee = analytics["weekly_assignee_resolution"]
    top_areas = analytics["top_contributing_areas"]
    actions = analytics["actions"]
    resolution_type_dist = analytics.get("resolution_type_distribution", {})

    # ----- Main category → Subcategory hierarchy -----
    st.markdown("---")
    st.markdown("### 📁 Main category & subcategories")
    for main in MAIN_CATEGORIES:
        subs = main_counts.get(main, {})
        total = sum(subs.values())
        if total == 0:
            continue
        sub_lines = [f"**{s}**: {c}" for s, c in sorted(subs.items(), key=lambda x: -x[1])]
        with st.expander(f"**{main}** — {total} tickets", expanded=True):
            for line in sub_lines:
                st.markdown(line)
            if not sub_lines:
                st.markdown("No tickets in this category.")

    # ----- Weekly category distribution matrix -----
    st.markdown("---")
    st.markdown("### 📅 Weekly category distribution")
    if weekly_cat:
        weeks = sorted(weekly_cat.keys(), reverse=True)[:8]
        rows = []
        for w in weeks:
            row = {"Week": w}
            for main in MAIN_CATEGORIES:
                row[main] = weekly_cat[w].get(main, 0)
            rows.append(row)
        matrix_df = pd.DataFrame(rows).set_index("Week")
        fig_heat = px.imshow(
            matrix_df.T,
            labels=dict(x="Week", y="Main category", color="Tickets"),
            aspect="auto",
            color_continuous_scale="Blues",
            text_auto="d",
        )
        fig_heat.update_layout(height=280, margin=dict(t=20, b=20), xaxis_tickangle=-45)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Weekly breakdown will appear when tickets have **Created** dates.")

    # ----- Assignee resolution: last 2 weeks -----
    st.markdown("### 👤 Assignee resolution (last 2 weeks)")
    if weekly_assignee:
        weeks_sorted = sorted(weekly_assignee.keys(), reverse=True)[:2]
        assignee_last2 = defaultdict(int)
        for w in weeks_sorted:
            for assignee, count in weekly_assignee[w].items():
                assignee_last2[assignee] += count
        if assignee_last2:
            res_2w = pd.DataFrame([
                {"Assignee": a, "Resolved (last 2 weeks)": c}
                for a, c in sorted(assignee_last2.items(), key=lambda x: -x[1])
            ])
            fig_assignee = px.bar(
                res_2w,
                x="Assignee",
                y="Resolved (last 2 weeks)",
                title="Resolved count by assignee (last 2 weeks)",
                color="Resolved (last 2 weeks)",
                color_continuous_scale="Blues",
            )
            fig_assignee.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_assignee, use_container_width=True)
        else:
            st.info("No resolutions in the last 2 weeks. Add **Resolved** date column for weekly data.")
    else:
        team = analytics["team_load"]
        res_df = pd.DataFrame([
            {"Assignee": a, "Total": v["total"], "Resolved": v["resolved"]}
            for a, v in team.items()
        ])
        if not res_df.empty:
            fig_bar = px.bar(
                res_df,
                x="Assignee",
                y=["Resolved", "Total"],
                barmode="group",
                title="Tickets by assignee (Total vs Resolved). Add Resolved date for last-2-weeks view.",
            )
            fig_bar.update_layout(height=300)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Add **Resolved** date column in the sheet for last-2-weeks resolution by assignee.")

    # ----- Top contributing issue areas -----
    st.markdown("### 📈 Top contributing issue areas")
    top_df = pd.DataFrame(top_areas, columns=["Subcategory", "Count"])
    fig_top = px.bar(
        top_df.head(8),
        x="Count",
        y="Subcategory",
        orientation="h",
        color="Count",
        color_continuous_scale="Teal",
    )
    fig_top.update_layout(height=320, showlegend=False, margin=dict(l=120), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_top, use_container_width=True)

    # ----- Resolution type distribution (categorized) -----
    st.markdown("### 📋 Resolution type distribution")
    res_type_cat = analytics.get("resolution_type_categorized", {})
    res_type_by_cat = analytics.get("resolution_type_by_category", {})
    
    if res_type_cat:
        res_cat_df = pd.DataFrame(
            list(res_type_cat.items()),
            columns=["Resolution Type", "Count"],
        ).sort_values("Count", ascending=False)
        fig_res = px.bar(
            res_cat_df,
            x="Resolution Type",
            y="Count",
            title="Tickets by resolution type (Tech Fix / Enablement / Non Tech Fix)",
            color="Resolution Type",
            color_discrete_map={"Tech Fix": "#3b82f6", "Enablement": "#10b981", "Non Tech Fix": "#6b7280"},
        )
        fig_res.update_layout(height=300, showlegend=False, xaxis_tickangle=0)
        st.plotly_chart(fig_res, use_container_width=True)
        
        if res_type_by_cat:
            st.markdown("#### Resolution Type vs Category")
            res_cat_matrix = []
            for cat, types in res_type_by_cat.items():
                res_cat_matrix.append({
                    "Category": cat,
                    "Tech Fix": types.get("Tech Fix", 0),
                    "Enablement": types.get("Enablement", 0),
                    "Non Tech Fix": types.get("Non Tech Fix", 0),
                })
            res_cat_df_matrix = pd.DataFrame(res_cat_matrix)
            if not res_cat_df_matrix.empty:
                fig_matrix = px.bar(
                    res_cat_df_matrix,
                    x="Category",
                    y=["Tech Fix", "Enablement", "Non Tech Fix"],
                    barmode="group",
                    title="Resolution type breakdown by category",
                    color_discrete_map={"Tech Fix": "#3b82f6", "Enablement": "#10b981", "Non Tech Fix": "#6b7280"},
                )
                fig_matrix.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_matrix, use_container_width=True)
    elif resolution_type_dist:
        res_type_df = pd.DataFrame(
            list(resolution_type_dist.items()),
            columns=["Resolution Type", "Count"],
        ).sort_values("Count", ascending=False)
        fig_res = px.bar(
            res_type_df,
            x="Resolution Type",
            y="Count",
            title="Tickets by resolution type (from sheet)",
            color="Count",
            color_continuous_scale="Viridis",
        )
        fig_res.update_layout(height=300, showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("Add a **Resolution Type** (or **Resolution type**) column in your Google Sheet to see distribution here.")

    # ----- Actions to be taken -----
    st.markdown("---")
    st.markdown("### ⚡ Actions to be taken")
    for action in actions:
        st.markdown(f"- {action}")

    # ----- Email Analytics -----
    st.markdown("---")
    st.markdown("### 📧 Send Analytics Email")
    with st.expander("📧 Email Configuration", expanded=False):
        recipient_email = st.text_input("Recipient Email", value=os.getenv("DEFAULT_RECIPIENT_EMAIL", ""))
        if st.button("Send Analytics Email"):
            if not recipient_email:
                st.error("Please enter recipient email address.")
            else:
                with st.spinner("Generating top 3 summaries and sending email..."):
                    analytics_for_email = {**analytics}
                    analytics_for_email["top3_subcategory_summaries"] = get_top3_subcategory_summaries(df, analytics)
                    success, message = send_email(recipient_email, analytics_for_email)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        st.caption("Configure SMTP settings in .env: SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SENDER_EMAIL")

    # ----- Raw table & downloads -----
    with st.expander("📋 View all classified tickets"):
        st.dataframe(df, use_container_width=True)

    st.download_button("Download Classified CSV", df.to_csv(index=False), "security_compliance_classified.csv", "text/csv")
    st.download_button("Download Analytics JSON", json.dumps(analytics, indent=2), "analytics.json", "application/json")