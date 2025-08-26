
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import logging
import unicodedata
import re
import io
import traceback
import numpy as np

# PDF Parsing libraries
import pymupdf as fitz # PyMuPDF, Hapoalim & Credit Report
import pdfplumber # Leumi & Discount

from openai import OpenAI
from openai import APIError # Specific import for API errors

# --- Logging Setup ---
# Configures logging to show informational messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Page and Styling Configuration ---
st.set_page_config(layout="wide", page_title="יועץ פיננסי משולב", page_icon="🧩")

# --- Custom CSS for modern design ---
# This CSS block restyles Streamlit components to create a cleaner, card-based interface
# with a professional color scheme and better typography.
st.markdown("""
<style>
    /* Main app background and font */
    .main {
        background-color: #f0f2f6;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Card-like containers for content */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }

    /* Styling for buttons */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #1c8dff;
        background-color: #1c8dff;
        color: white;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #0073e6;
        border-color: #0073e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(28, 141, 255, 0.5) !important;
    }

    /* Specific styling for secondary/navigation buttons */
    .stButton[key*="prev"] > button, .stButton[key*="skip"] > button {
        background-color: #f0f2f6;
        color: #333;
        border: 1px solid #ccc;
    }
    .stButton[key*="prev"] > button:hover, .stButton[key*="skip"] > button:hover {
        background-color: #e6e6e6;
        border-color: #bbb;
    }

    /* Header styling */
    h1, h2, h3 {
        color: #1a2a6c; /* Dark blue for headers */
    }

    /* Metric styling */
    .stMetric {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e0e0e0;
    }

    /* Chat input styling */
    .stChatInput {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)


# --- OpenAI Client Setup ---
client = None # Initialize client to None
try:
    # Attempt to get API key from secrets
    api_key = st.secrets.get("OPENAI_API_KEY")
    if api_key: # Check if key exists and is not empty
        client = OpenAI(api_key=api_key)
        logging.info("OpenAI client initialized successfully.")
    else:
        logging.warning("OPENAI_API_KEY not found or is empty in Streamlit secrets.")
        # Do not show an error here initially, only in the chat section if the client is needed.

except Exception as e:
    logging.error(f"Error loading OpenAI API key or initializing client: {e}", exc_info=True)
    # This error will show up if the secrets handling itself fails.
    st.error(f"שגיאה בטעינת מפתח OpenAI. הצ'אטבוט עשוי לא לפעול כראוי.")


# --- Helper Functions (General Purpose) ---
def clean_number_general(text):
    """Cleans numeric strings, handling currency symbols, commas, and parentheses."""
    if text is None: return None
    text = str(text).strip()
    text = re.sub(r'[₪,]', '', text)
    if text.startswith('(') and text.endswith(')'): text = '-' + text[1:-1]
    if text.endswith('-'): text = '-' + text[:-1]
    try:
        if text == "": return None # Handle empty string after cleaning
        return float(text)
    except ValueError:
        logging.debug(f"Could not convert '{text}' to float.")
        return None

def parse_date_general(date_str):
    """Parses date strings in multiple formats."""
    if date_str is None or pd.isna(date_str) or not isinstance(date_str, str): return None
    date_str = date_str.strip()
    if not date_str: return None
    for fmt in ('%d/%m/%Y', '%d/%m/%y'):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            pass
    logging.debug(f"Could not parse date: {date_str}")
    return None

def normalize_text_general(text):
    """Normalizes Unicode text (removes potential hidden chars, ensures NFC)."""
    if text is None: return None
    text = str(text).replace('\r', ' ').replace('\n', ' ').replace('\u200b', '').strip()
    return unicodedata.normalize('NFC', text)

# --- PDF Parsers ---

# --- HAPOALIM PARSER ---
def extract_transactions_from_pdf_hapoalim(pdf_content_bytes, filename_for_logging="hapoalim_pdf"):
    """Extracts Date and Balance from Hapoalim PDF based on line patterns."""
    transactions = []
    try:
        doc = fitz.open(stream=pdf_content_bytes, filetype="pdf")
    except Exception as e:
        logging.error(f"Hapoalim: Failed to open/process PDF {filename_for_logging}: {e}", exc_info=True)
        return pd.DataFrame()


    date_pattern_end = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})\s*$")
    balance_pattern_start = re.compile(r"^\s*(₪?-?[\d,]+\.\d{2})")

    logging.info(f"Starting Hapoalim PDF parsing for {filename_for_logging}")

    for page_num, page in enumerate(doc):
        try:
            lines = page.get_text("text", sort=True).splitlines()
            for line_num, line_text in enumerate(lines):
                original_line = line_text
                line_normalized = normalize_text_general(original_line)

                if not line_normalized or len(line_normalized) < 10: continue

                date_match = date_pattern_end.search(original_line)
                if date_match:
                    date_str = date_match.group(1)
                    parsed_date = parse_date_general(date_str)

                    if parsed_date:
                        balance_match = balance_pattern_start.search(original_line)
                        if balance_match:
                            balance_str = balance_match.group(1)
                            balance = clean_number_general(balance_str)

                            if balance is not None:
                                lower_line = line_normalized.lower()
                                if "יתרה לסוף יום" in lower_line or "עובר ושב" in lower_line or "תנועות בחשבון" in lower_line or "עמוד" in lower_line or "סך הכל" in lower_line or "הודעה זו כוללת" in lower_line:
                                    logging.debug(f"Hapoalim: Skipping potential header/footer/summary line: {original_line.strip()}")
                                    continue

                                transactions.append({
                                    'Date': parsed_date,
                                    'Balance': balance,
                                })
        except Exception as e:
            logging.error(f"Hapoalim: Error processing line {line_num+1} on page {page_num+1}: {e}", exc_info=True)
            continue

    doc.close()

    if not transactions:
        logging.warning(f"Hapoalim: No transactions found in {filename_for_logging}")
        return pd.DataFrame()

    df = pd.DataFrame(transactions)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
    df = df.dropna(subset=['Date', 'Balance'])

    df = df.sort_values(by='Date').groupby('Date')['Balance'].last().reset_index()
    df = df.sort_values(by='Date').reset_index(drop=True)

    logging.info(f"Hapoalim: Successfully extracted {len(df)} unique balance points from {filename_for_logging}")
    return df[['Date', 'Balance']]

# --- LEUMI PARSER ---
def extract_leumi_transactions_line_by_line(pdf_content_bytes, filename_for_logging="leumi_pdf"):
    transactions_data = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_content_bytes)) as pdf:
            previous_balance = None
            first_transaction_processed = False
            logging.info(f"Starting Leumi PDF parsing for {filename_for_logging}")

            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2, layout=True)
                    if not text: continue

                    lines = text.splitlines()
                    for line_num, line_text in enumerate(lines):
                        normalized_line = normalize_text_general(line_text.strip())
                        if not normalized_line: continue
                        
                        # Simplified pattern for Leumi based on common structures
                        pattern = re.compile(r"^(.*?)\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+([\d,\.-]+)\s+([\d,\.-]+)\s+([\d,\.-]+)$")
                        match = pattern.match(normalized_line)

                        if match:
                            date_str = match.group(2)
                            balance_str = match.group(6)
                            
                            parsed_date = parse_date_general(date_str)
                            balance = clean_number_general(balance_str)

                            if parsed_date and balance is not None:
                                transactions_data.append({'Date': parsed_date, 'Balance': balance})
                                logging.debug(f"Leumi: Found transaction - Date: {parsed_date}, Balance: {balance}")

                except Exception as e:
                     logging.error(f"Leumi: Error processing line {line_num+1} on page {page_num+1}: {e}", exc_info=True)

    except Exception as e:
        logging.error(f"Leumi: FATAL ERROR processing PDF {filename_for_logging}: {e}", exc_info=True)
        return pd.DataFrame()

    if not transactions_data:
        logging.warning(f"Leumi: No transaction balances found in {filename_for_logging}")
        return pd.DataFrame()

    df = pd.DataFrame(transactions_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
    df = df.dropna(subset=['Date', 'Balance'])
    df = df.sort_values(by='Date').groupby('Date')['Balance'].last().reset_index()
    
    logging.info(f"Leumi: Successfully extracted {len(df)} unique balance points from {filename_for_logging}")
    return df[['Date', 'Balance']]

# --- DISCOUNT PARSER ---
def extract_and_parse_discont_pdf(pdf_content_bytes, filename_for_logging="discount_pdf"):
    transactions = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_content_bytes)) as pdf:
            logging.info(f"Starting Discount PDF parsing for {filename_for_logging}")
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    lines = text.splitlines()
                    for line in lines:
                        # Pattern: date, date, description, ref, amount, balance
                        match = re.search(r"(\d{2}/\d{2}/\d{2,4})\s+(\d{2}/\d{2}/\d{2,4}).*?([\d,\.-]+)\s*$", line)
                        if match:
                            date_str = match.group(1)
                            balance_str = match.group(3)
                            
                            parsed_date = parse_date_general(date_str)
                            balance = clean_number_general(balance_str)

                            if parsed_date and balance is not None:
                                transactions.append({'Date': parsed_date, 'Balance': balance})
    except Exception as e:
        logging.error(f"Discount: FATAL ERROR processing PDF {filename_for_logging}: {e}", exc_info=True)
        return pd.DataFrame()

    if not transactions:
        logging.warning(f"Discount: No transaction balances found in {filename_for_logging}")
        return pd.DataFrame()

    df = pd.DataFrame(transactions)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
    df = df.dropna(subset=['Date', 'Balance'])
    df = df.sort_values(by='Date').groupby('Date')['Balance'].last().reset_index()

    logging.info(f"Discount: Successfully extracted {len(df)} unique balance points from {filename_for_logging}")
    return df[['Date', 'Balance']]

# --- NEW AND IMPROVED CREDIT REPORT PARSER ---
def extract_credit_data_with_pdfplumber(pdf_content_bytes, filename_for_logging="credit_report_pdf"):
    """
    Extracts structured credit data from a report PDF using pdfplumber's table extraction.
    This method is more robust for tabular data like the official credit report.
    """
    extracted_rows = []
    logging.info(f"Starting Credit Report PDF parsing with pdfplumber for {filename_for_logging}")

    try:
        with pdfplumber.open(io.BytesIO(pdf_content_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if not tables:
                    continue

                for table in tables:
                    if not table or len(table) < 2: continue # Skip empty or header-only tables
                    
                    header = [h.replace('\n', ' ') if h else '' for h in table[0]]
                    col_map = {}
                    current_section = "לא ידוע"

                    # Identify the table type and map columns
                    if 'גובה מסגרת' in header and 'יתרת חוב' in header:
                        current_section = 'חשבון עובר ושב'
                        col_map = {'שם מקור המידע המדווח': 'שם בנק/מקור', 'גובה מסגרת': 'גובה מסגרת', 'יתרת חוב': 'יתרת חוב', 'יתרה שלא שולמה במועד': 'יתרה שלא שולמה'}
                    elif 'סכום הלוואות מקורי' in header:
                        current_section = 'הלוואה'
                        col_map = {'שם מקור המידע המדווח': 'שם בנק/מקור', 'סכום הלוואות מקורי': 'סכום מקורי', 'יתרת חוב': 'יתרת חוב', 'יתרה שלא שולמה במועד': 'יתרה שלא שולמה'}
                    elif 'גובה מסגרות' in header:
                        current_section = 'מסגרת אשראי מתחדשת'
                        col_map = {'שם מקור המידע המדווח': 'שם בנק/מקור', 'גובה מסגרות': 'גובה מסגרת', 'יתרת חוב': 'יתרת חוב', 'יתרה שלא שולמה במועד': 'יתרה שלא שולמה'}
                    else:
                        continue # Not a table we recognize

                    # Create a reverse map from header name to index for safe access
                    header_map = {name: i for i, name in enumerate(header)}

                    # Process rows in the identified table (skip header)
                    for row_cells in table[1:]:
                        # Skip summary rows
                        if row_cells[0] and 'סה"כ' in row_cells[0]: continue
                        
                        entry = {"סוג עסקה": current_section}
                        
                        for header_name, key_name in col_map.items():
                            if header_name in header_map:
                                col_index = header_map[header_name]
                                raw_value = row_cells[col_index]
                                if key_name == 'שם בנק/מקור':
                                    entry[key_name] = normalize_text_general(raw_value)
                                else:
                                    entry[key_name] = clean_number_general(raw_value)
                        
                        # Fill missing columns with defaults
                        entry.setdefault('גובה מסגרת', np.nan)
                        entry.setdefault('סכום מקורי', np.nan)
                        entry.setdefault('יתרת חוב', np.nan)
                        entry.setdefault('יתרה שלא שולמה', 0.0)

                        extracted_rows.append(entry)
                        logging.debug(f"CR: Appended row: {entry}")

    except Exception as e:
        logging.error(f"CreditReport (pdfplumber): FATAL ERROR processing {filename_for_logging}: {e}", exc_info=True)
        return pd.DataFrame()

    if not extracted_rows:
        logging.warning(f"CreditReport (pdfplumber): No structured entries found in {filename_for_logging}")
        return pd.DataFrame()

    df = pd.DataFrame(extracted_rows)
    
    final_cols = ["סוג עסקה", "שם בנק/מקור", "גובה מסגרת", "סכום מקורי", "יתרת חוב", "יתרה שלא שולמה"]
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[final_cols]
    
    for col in ["גובה מסגרת", "סכום מקורי", "יתרת חוב", "יתרה שלא שולמה"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['יתרה שלא שולמה'] = df['יתרה שלא שולמה'].fillna(0)

    df = df.dropna(subset=['גובה מסגרת', 'סכום מקורי', 'יתרת חוב'], how='all').reset_index(drop=True)

    logging.info(f"CreditReport (pdfplumber): Successfully extracted {len(df)} entries from {filename_for_logging}")
    return df

# --- Initialize Session State ---
def initialize_session_state():
    defaults = {
        'app_stage': "welcome",
        'questionnaire_stage': 0,
        'answers': {},
        'classification_details': {},
        'chat_messages': [],
        'df_bank_uploaded': pd.DataFrame(),
        'df_credit_uploaded': pd.DataFrame(),
        'bank_type_selected': "ללא דוח בנק",
        'total_debt_from_credit_report': None,
        'uploaded_bank_file_name': None,
        'uploaded_credit_file_name': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

def reset_all_data():
    """Resets all session state variables to their initial state for a fresh start."""
    logging.info("Resetting all application data.")
    st.session_state.clear()
    initialize_session_state()
    st.rerun()

# --- Streamlit App Layout ---
st.title("🧩 יועץ פיננסי משולב")
st.markdown("### ניתוח מצב פיננסי, סיווג וייעוץ אישי")

# --- Sidebar ---
with st.sidebar:
    st.header("אפשרויות ניווט")
    if st.button("🔄 התחל מחדש", use_container_width=True):
        reset_all_data()
    st.divider()
    st.info("כלי זה נועד למטרות אבחון ראשוני ואינו מהווה ייעוץ פיננסי מקצועי.")
    st.caption("© כל הזכויות שמורות.")

# --- Main Application Flow ---

# --- WELCOME STAGE ---
if st.session_state.app_stage == "welcome":
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("👋 ברוכים הבאים ליועץ הפיננסי המשולב!")
        st.markdown("""
        כלי זה יסייע לכם לקבל תמונה רחבה על מצבכם הפיננסי. התהליך משלב ניתוח דוחות אוטומטי עם שאלון מקיף להבנת ההקשר המלא.
        
        **שלבי התהליך:**
        1.  **העלאת דוחות (מומלץ):** דוח יתרות ועסקאות מהבנק ודוח נתוני אשראי.
        2.  **שאלון פיננסי:** מילוי פרטים על הכנסות, הוצאות, חובות ומצבכם הכללי.
        3.  **סיכום וניתוח:** קבלת סיווג פיננסי, המלצות ראשוניות, תרשימים ושיחה עם יועץ וירטואלי.
        """)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 התחל בהעלאת קבצים (מומלץ)", key="start_with_files", use_container_width=True):
                st.session_state.app_stage = "file_upload"
                st.rerun()
        with col2:
            if st.button("✍️ התחל ישר מהשאלון", key="start_with_questionnaire", use_container_width=True):
                st.session_state.app_stage = "questionnaire"
                st.session_state.questionnaire_stage = 0
                st.session_state.answers = {}
                st.session_state.chat_messages = []
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- FILE UPLOAD STAGE ---
elif st.session_state.app_stage == "file_upload":
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("שלב 1: העלאת דוחות")
        st.markdown("העלאת דוחות מאפשרת ניתוח מדויק ומעמיק יותר. המידע מעובד באופן מאובטח ולא נשמר.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📄 דוח בנק")
            bank_type_options = ["ללא דוח בנק", "הפועלים", "דיסקונט", "לאומי"]
            st.session_state.bank_type_selected = st.selectbox(
                "בחר את סוג דוח הבנק:", bank_type_options, key="bank_type_selector_main"
            )
            uploaded_bank_file = None
            if st.session_state.bank_type_selected != "ללא דוח בנק":
                uploaded_bank_file = st.file_uploader(f"העלה קובץ PDF של דוח {st.session_state.bank_type_selected}", type="pdf", key="bank_pdf_uploader_main")
                if uploaded_bank_file and st.session_state.get('uploaded_bank_file_name') != uploaded_bank_file.name:
                    st.session_state.df_bank_uploaded = pd.DataFrame()
                    st.session_state.uploaded_bank_file_name = uploaded_bank_file.name
                if st.session_state.uploaded_bank_file_name:
                    st.success(f"קובץ בנק '{st.session_state.uploaded_bank_file_name}' נטען.")
        with col2:
            st.subheader("💳 דוח נתוני אשראי")
            uploaded_credit_file = st.file_uploader("העלה קובץ PDF של דוח נתוני אשראי (מומלץ)", type="pdf", key="credit_pdf_uploader_main")
            if uploaded_credit_file and st.session_state.get('uploaded_credit_file_name') != uploaded_credit_file.name:
                 st.session_state.df_credit_uploaded = pd.DataFrame()
                 st.session_state.total_debt_from_credit_report = None
                 st.session_state.uploaded_credit_file_name = uploaded_credit_file.name
            if st.session_state.uploaded_credit_file_name:
                st.success(f"קובץ אשראי '{st.session_state.uploaded_credit_file_name}' נטען.")
        st.divider()
        if st.button("🔬 עבד קבצים והמשך לשאלון", key="process_files_button", use_container_width=True):
            with st.spinner("מעבד קבצים... אנא המתן/י..."):
                if uploaded_bank_file and st.session_state.bank_type_selected != "ללא דוח בנק":
                    try:
                        parser_map = {
                            "הפועלים": extract_transactions_from_pdf_hapoalim,
                            "לאומי": extract_leumi_transactions_line_by_line,
                            "דיסקונט": extract_and_parse_discont_pdf
                        }
                        parser_func = parser_map.get(st.session_state.bank_type_selected)
                        if parser_func:
                            st.session_state.df_bank_uploaded = parser_func(uploaded_bank_file.getvalue(), uploaded_bank_file.name)
                        if st.session_state.df_bank_uploaded.empty:
                            st.warning(f"לא חולצו נתונים מדוח הבנק ({st.session_state.bank_type_selected}).")
                    except Exception as e:
                        logging.error(f"Error processing bank file {uploaded_bank_file.name}: {e}", exc_info=True)
                        st.error(f"שגיאה בעיבוד דוח הבנק: {e}")
                if uploaded_credit_file:
                    try:
                        # --- THIS IS THE UPDATED FUNCTION CALL ---
                        st.session_state.df_credit_uploaded = extract_credit_data_with_pdfplumber(uploaded_credit_file.getvalue(), uploaded_credit_file.name)
                        if st.session_state.df_credit_uploaded.empty:
                            st.warning("לא חולצו נתונים מדוח האשראי.")
                        elif 'יתרת חוב' in st.session_state.df_credit_uploaded.columns:
                            total_debt = st.session_state.df_credit_uploaded['יתרת חוב'].fillna(0).sum()
                            st.session_state.total_debt_from_credit_report = total_debt
                    except Exception as e:
                        logging.error(f"Error processing credit file {uploaded_credit_file.name}: {e}", exc_info=True)
                        st.error(f"שגיאה בעיבוד דוח האשראי: {e}")
            st.success("עיבוד הקבצים הסתיים!")
            st.session_state.app_stage = "questionnaire"
            st.session_state.questionnaire_stage = 0
            st.session_state.chat_messages = []
            st.rerun()
        if st.button("דלג והמשך לשאלון", key="skip_files_button", use_container_width=True):
            st.session_state.df_bank_uploaded, st.session_state.df_credit_uploaded = pd.DataFrame(), pd.DataFrame()
            st.session_state.total_debt_from_credit_report = None
            st.session_state.app_stage = "questionnaire"
            st.session_state.questionnaire_stage = 0
            st.session_state.chat_messages = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- QUESTIONNAIRE STAGE ---
elif st.session_state.app_stage == "questionnaire":
    st.header("שלב 2: שאלון פיננסי")
    st.markdown("אנא ענה/י על השאלות בכנות כדי לקבל ניתוח מדויק ככל האפשר.")
    total_stages = 4
    current_stage = st.session_state.questionnaire_stage
    progress_value = 1.0 if current_stage >= 100 else (current_stage + 1) / total_stages
    st.progress(progress_value, text=f"שלב {current_stage+1 if current_stage < 100 else total_stages} מתוך {total_stages}")
    q_stage = st.session_state.questionnaire_stage

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if q_stage == 0:
            st.subheader("חלק א': שאלות פתיחה")
            st.session_state.answers['q1_unusual_event'] = st.text_area("1. האם קרה אירוע חריג שבעקבותיו פניתם לייעוץ?", value=st.session_state.answers.get('q1_unusual_event', ''))
            st.session_state.answers['q2_other_funding'] = st.text_area("2. האם בדקתם מקורות מימון או פתרונות אחרים?", value=st.session_state.answers.get('q2_other_funding', ''))
            st.session_state.answers['q3_existing_loans_bool_radio'] = st.radio("3. האם קיימות הלוואות נוספות (לא משכנתא)?", ("כן", "לא"), index=("כן", "לא").index(st.session_state.answers.get('q3_existing_loans_bool_radio', 'לא')))
            if st.session_state.answers['q3_existing_loans_bool_radio'] == "כן":
                st.session_state.answers['q3_loan_repayment_amount'] = st.number_input("מה גובה ההחזר החודשי הכולל עליהן?", min_value=0.0, value=float(st.session_state.answers.get('q3_loan_repayment_amount', 0.0)), step=100.0)
            else:
                st.session_state.answers['q3_loan_repayment_amount'] = 0.0
            st.session_state.answers['q4_financially_balanced_bool_radio'] = st.radio("4. האם אתם מאוזנים כלכלית (הכנסות מכסות הוצאות)?", ("כן", "בערך", "לא"), index=("כן", "בערך", "לא").index(st.session_state.answers.get('q4_financially_balanced_bool_radio', 'כן')))
            st.session_state.answers['q4_situation_change_next_year'] = st.text_area("האם מצבכם הכלכלי צפוי להשתנות משמעותית בשנה הקרובה?", value=st.session_state.answers.get('q4_situation_change_next_year', ''))
        elif q_stage == 1:
            st.subheader("חלק ב': הכנסות (נטו חודשי)")
            st.session_state.answers['income_employee'] = st.number_input("הכנסתך (נטו):", min_value=0.0, value=float(st.session_state.answers.get('income_employee', 0.0)), step=100.0)
            st.session_state.answers['income_partner'] = st.number_input("הכנסת בן/בת הזוג (נטו):", min_value=0.0, value=float(st.session_state.answers.get('income_partner', 0.0)), step=100.0)
            st.session_state.answers['income_other'] = st.number_input("הכנסות נוספות (קצבאות, שכירות וכו'):", min_value=0.0, value=float(st.session_state.answers.get('income_other', 0.0)), step=100.0)
            total_net_income = st.session_state.answers.get('income_employee', 0.0) + st.session_state.answers.get('income_partner', 0.0) + st.session_state.answers.get('income_other', 0.0)
            st.session_state.answers['total_net_income'] = total_net_income
            st.metric("💰 סך הכנסות נטו (חודשי):", f"{total_net_income:,.0f} ₪")
        elif q_stage == 2:
            st.subheader("חלק ג': הוצאות קבועות חודשיות")
            st.session_state.answers['expense_rent_mortgage'] = st.number_input("שכירות / החזר משכנתא:", min_value=0.0, value=float(st.session_state.answers.get('expense_rent_mortgage', 0.0)), step=100.0)
            default_debt_repayment = float(st.session_state.answers.get('q3_loan_repayment_amount', 0.0))
            st.session_state.answers['expense_debt_repayments'] = st.number_input("החזרי הלוואות נוספות:", min_value=0.0, value=float(st.session_state.answers.get('expense_debt_repayments', default_debt_repayment)), step=100.0)
            st.session_state.answers['expense_alimony_other'] = st.number_input("מזונות / הוצאות קבועות גדולות אחרות:", min_value=0.0, value=float(st.session_state.answers.get('expense_alimony_other', 0.0)), step=100.0)
            total_fixed_expenses = st.session_state.answers.get('expense_rent_mortgage', 0.0) + st.session_state.answers.get('expense_debt_repayments', 0.0) + st.session_state.answers.get('expense_alimony_other', 0.0)
            st.session_state.answers['total_fixed_expenses'] = total_fixed_expenses
            total_net_income = float(st.session_state.answers.get('total_net_income', 0.0))
            monthly_balance = total_net_income - total_fixed_expenses
            st.metric("💸 סך הוצאות קבועות:", f"{total_fixed_expenses:,.0f} ₪")
            st.metric("📊 יתרה פנויה חודשית:", f"{monthly_balance:,.0f} ₪", delta_color="inverse")
            if monthly_balance < 0: st.warning("שימו לב: ההוצאות הקבועות גבוהות מההכנסות.")
        elif q_stage == 3:
            st.subheader("חלק ד': חובות ופיגורים")
            default_total_debt = st.session_state.total_debt_from_credit_report if st.session_state.total_debt_from_credit_report is not None else float(st.session_state.answers.get('total_debt_amount', 0.0))
            if st.session_state.total_debt_from_credit_report is not None:
                st.info(f"סך יתרת החוב שחושבה מדוח האשראי הוא: {st.session_state.total_debt_from_credit_report:,.0f} ₪. ניתן לעדכן את הסכום.")
            else:
                st.info("אנא הזן/י את סך כל החובות הקיימים (למעט משכנתא).")
            st.session_state.answers['total_debt_amount'] = st.number_input("מה היקף החובות הכולל (למעט משכנתא)?", min_value=0.0, value=default_total_debt, step=100.0)
            st.session_state.answers['arrears_collection_proceedings_radio'] = st.radio("האם קיימים פיגורים משמעותיים בתשלומים או הליכי גבייה?",("כן", "לא"), index=("כן", "לא").index(st.session_state.answers.get('arrears_collection_proceedings_radio', 'לא')))
        elif q_stage == 100:
            st.subheader("שאלות הבהרה נוספות")
            st.warning(f"תוצאה ראשונית: יחס החוב להכנסה שלך הוא {st.session_state.answers.get('debt_to_income_ratio', 0.0):.2f}. ({st.session_state.classification_details.get('description')})")
            if st.session_state.answers.get('arrears_collection_proceedings_radio', 'לא') == 'כן':
                 st.error("זוהו הליכי גבייה. מצב זה מסווג אוטומטית כ'אדום'.")
            else:
                total_debt = float(st.session_state.answers.get('total_debt_amount', 0.0))
                fifty_percent_debt = total_debt * 0.5
                st.session_state.answers['can_raise_50_percent_radio'] = st.radio(f"האם תוכל/י לגייס סכום של כ-50% מהחוב ({fifty_percent_debt:,.0f} ₪) ממקורות תמיכה?",("כן", "לא"), index=("כן", "לא").index(st.session_state.answers.get('can_raise_50_percent_radio', 'לא')))
        st.divider()
        cols = st.columns([1, 5, 1])
        if q_stage > 0:
            if cols[0].button("⬅️ הקודם", key=f"q_s{q_stage}_prev", use_container_width=True):
                if q_stage == 100: st.session_state.questionnaire_stage = 3
                else: st.session_state.questionnaire_stage -= 1
                st.rerun()
        if q_stage < 3:
            if cols[2].button("הבא ➡️", key=f"q_s{q_stage}_next", use_container_width=True):
                st.session_state.questionnaire_stage += 1
                st.rerun()
        elif q_stage == 3:
            if cols[2].button("✅ סיום וקבלת סיכום", key="q_s3_next_finish", use_container_width=True):
                total_debt = float(st.session_state.answers.get('total_debt_amount', 0.0))
                net_income = float(st.session_state.answers.get('total_net_income', 0.0))
                annual_income = net_income * 12
                st.session_state.answers['annual_income'] = annual_income
                ratio = (total_debt / annual_income) if annual_income > 0 else float('inf') if total_debt > 0 else 0.0
                st.session_state.answers['debt_to_income_ratio'] = ratio
                arrears = st.session_state.answers.get('arrears_collection_proceedings_radio', 'לא') == 'כן'
                if arrears:
                    st.session_state.classification_details = {'classification': "אדום", 'description': "קיימים פיגורים או הליכי גבייה.", 'color': "red"}
                    st.session_state.app_stage = "summary"
                elif ratio < 1:
                    st.session_state.classification_details = {'classification': "ירוק", 'description': "יחס חוב להכנסה נמוך משנת הכנסה.", 'color': "green"}
                    st.session_state.app_stage = "summary"
                elif 1 <= ratio <= 2:
                    st.session_state.classification_details = {'classification': "צהוב (בבדיקה)", 'description': "יחס חוב להכנסה הוא בין 1-2 שנות הכנסה.", 'color': "orange"}
                    st.session_state.questionnaire_stage = 100
                else: # ratio > 2
                    st.session_state.classification_details = {'classification': "אדום", 'description': "יחס חוב להכנסה גבוה משנתיים הכנסה.", 'color': "red"}
                    st.session_state.app_stage = "summary"
                st.rerun()
        elif q_stage == 100:
             if cols[2].button("המשך לסיכום ➡️", key="q_s100_to_summary", use_container_width=True):
                arrears = st.session_state.answers.get('arrears_collection_proceedings_radio', 'לא') == 'כן'
                can_raise_funds = st.session_state.answers.get('can_raise_50_percent_radio', 'לא') == 'כן'
                if arrears:
                    st.session_state.classification_details.update({'classification': "אדום", 'description': "יחס חוב בינוני אך קיימים הליכי גבייה.", 'color': "red"})
                elif can_raise_funds:
                    st.session_state.classification_details.update({'classification': "צהוב", 'description': "יחס חוב בינוני, אך קיימת יכולת גיוס הון עצמי.", 'color': "orange"})
                else:
                    st.session_state.classification_details.update({'classification': "אדום", 'description': "יחס חוב בינוני וללא יכולת גיוס הון עצמי.", 'color': "red"})
                st.session_state.app_stage = "summary"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- SUMMARY & ANALYSIS STAGE ---
elif st.session_state.app_stage == "summary":
    st.header("שלב 3: סיכום, ניתוח וייעוץ")
    st.markdown("להלן סיכום הנתונים שאספנו, ניתוח ראשוני והמלצות להמשך.")

    # Retrieve metrics for summary
    answers = st.session_state.answers
    total_net_income = float(answers.get('total_net_income', 0.0))
    total_fixed_expenses = float(answers.get('total_fixed_expenses', 0.0))
    monthly_balance = total_net_income - total_fixed_expenses
    total_debt = float(answers.get('total_debt_amount', 0.0))
    annual_income = float(answers.get('annual_income', 0.0))
    debt_ratio = float(answers.get('debt_to_income_ratio', 0.0))
    classification_details = st.session_state.classification_details
    
    # Define dataframes early for use in multiple places
    df_bank = st.session_state.df_bank_uploaded.dropna(subset=['Date', 'Balance']).sort_values('Date')
    df_credit = st.session_state.df_credit_uploaded.copy()

    # --- Classification and Recommendations Card ---
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("סיווג והמלצה ראשונית")
        color = classification_details.get('color', 'gray')
        classification = classification_details.get('classification', 'לא נקבע')

        if color == "green":
            st.success(f"🟢 **סיווג: {classification}**")
            st.markdown("""
            **מצב יציב.** יחס החוב להכנסה נמוך ומאפשר גמישות פיננסית.
            * **המלצה:** המשך/י בניהול פיננסי אחראי. זהו זמן טוב לשקול הגדלת חיסכון או השקעות.
            """)
        elif color == "orange":
            st.warning(f"🟡 **סיווג: {classification}**")
            st.markdown(f"""
            **מצב הדורש בדיקה ותשומת לב.** {classification_details.get('description', '')}
            * **המלצה:** יש לבחון לעומק את פירוט החובות וההוצאות. מומלץ לבנות תוכנית פעולה לצמצום החובות, תוך בחינת אפשרויות להגדלת הכנסה או קיצוץ בהוצאות.
            """)
        elif color == "red":
            st.error(f"🔴 **סיווג: {classification}**")
            st.markdown(f"""
            **מצב הדורש התערבות מיידית.** {classification_details.get('description', '')}
            * **המלצה:** יש לפנות בהקדם לייעוץ מקצועי בכלכלת המשפחה. חשוב להבין את היקף החוב המלא, להפסיק לצבור חוב חדש ולבנות תוכנית חירום.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Financial Summary Metrics Card ---
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 סיכום נתונים פיננסיים")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 סך הכנסות נטו (חודשי)", f"{total_net_income:,.0f} ₪")
            st.metric("💸 סך הוצאות קבועות (חודשי)", f"{total_fixed_expenses:,.0f} ₪")
        with col2:
            st.metric("📊 יתרה פנויה (חודשי)", f"{monthly_balance:,.0f} ₪")
            st.metric("🏦 סך חובות (ללא משכנתא)", f"{total_debt:,.0f} ₪")
            if st.session_state.total_debt_from_credit_report is not None:
                st.caption(f"(מדוח אשראי: {st.session_state.total_debt_from_credit_report:,.0f} ₪)")
        with col3:
            st.metric("📈 הכנסה שנתית", f"{annual_income:,.0f} ₪")
            st.metric("⚖️ יחס חוב להכנסה שנתית", f"{debt_ratio:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Visualizations ---
    st.subheader("🎨 ויזואליזציות מרכזיות")
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Debt Breakdown Pie (Altair Donut)
            if not df_credit.empty and 'יתרת חוב' in df_credit.columns:
                df_credit['יתרת חוב'] = pd.to_numeric(df_credit['יתרת חוב'], errors='coerce').fillna(0)
                debt_summary = df_credit.groupby("סוג עסקה")["יתרת חוב"].sum().reset_index()
                debt_summary = debt_summary[debt_summary['יתרת חוב'] > 0]
                if not debt_summary.empty:
                    chart_pie = alt.Chart(debt_summary).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta(field="יתרת חוב", type="quantitative"),
                        color=alt.Color(field="סוג עסקה", type="nominal", title="סוג עסקה"),
                        tooltip=["סוג עסקה", alt.Tooltip("יתרת חוב", format=',.0f')]
                    ).properties(title="פירוט חובות (מדוח אשראי)")
                    st.altair_chart(chart_pie, use_container_width=True)
                else:
                    st.info("לא נמצאו נתוני חוב משמעותיים בדוח האשראי.")
            else:
                st.info("לא הועלה דוח נתוני אשראי לפירוט חובות.")
            st.markdown('</div>', unsafe_allow_html=True)

    with viz_col2:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Debt vs. Income Bar (Altair)
            if total_debt > 0 or annual_income > 0:
                comparison_data = pd.DataFrame({
                    'קטגוריה': ['סך חובות', 'הכנסה שנתית'],
                    'סכום': [total_debt, annual_income]
                })
                chart_bar = alt.Chart(comparison_data).mark_bar().encode(
                    x=alt.X('קטגוריה', sort=None, title=None),
                    y=alt.Y('סכום', title="סכום ב-₪"),
                    color=alt.Color('קטגוריה', legend=None),
                    tooltip=['קטגוריה', alt.Tooltip('סכום', format=',.0f')]
                ).properties(title="השוואת חובות להכנסה שנתית")
                st.altair_chart(chart_bar, use_container_width=True)
            else:
                st.info("אין נתוני חוב או הכנסה להצגת השוואה.")
            st.markdown('</div>', unsafe_allow_html=True)

    # Bank Balance Trend (Altair Line Chart)
    if not df_bank.empty:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            chart_line = alt.Chart(df_bank).mark_line(point=True).encode(
                x=alt.X('Date:T', title="תאריך"),
                y=alt.Y('Balance:Q', title="יתרה ב-₪"),
                tooltip=[alt.Tooltip('Date', title='תאריך'), alt.Tooltip('Balance', title='יתרה', format=',.0f')]
            ).properties(title="מגמת יתרת חשבון הבנק").interactive()
            st.altair_chart(chart_line, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Chatbot and Raw Data Card ---
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        with st.expander("הצג נתונים מפורטים שחולצו מהדוחות"):
            if not df_credit.empty:
                st.write("נתוני דוח אשראי:")
                st.dataframe(df_credit.style.format(precision=0, thousands=","), use_container_width=True)
            if not df_bank.empty:
                st.write(f"נתוני דוח בנק ({st.session_state.bank_type_selected}):")
                st.dataframe(df_bank.style.format({"Balance": '{:,.2f}'}), use_container_width=True)

        st.divider()
        
        st.header("💬 צ'אט עם יועץ פיננסי וירטואלי")
        if client:
            st.markdown("כעת תוכל/י לשאול שאלות על מצבך, לבקש הבהרות על הניתוח, או לקבל רעיונות לצעדים הבאים.")
            
            # Prepare context for chatbot
            financial_context_parts = [
                f"- סיווג פיננסי: {classification} ({classification_details.get('description', '')})",
                f"- הכנסה חודשית נטו: {total_net_income:,.0f} ₪",
                f"- הוצאות קבועות חודשיות: {total_fixed_expenses:,.0f} ₪",
                f"- יתרה פנויה חודשית: {monthly_balance:,.0f} ₪",
                f"- סך חובות (ללא משכנתא): {total_debt:,.0f} ₪",
                f"- יחס חוב להכנסה שנתית: {debt_ratio:.2%}"
            ]
            if not df_credit.empty:
                financial_context_parts.append("\nפירוט חובות מדוח אשראי:")
                for _, row in df_credit.head(10).iterrows():
                    financial_context_parts.append(f"  - {row.get('סוג עסקה', '')} ב{row.get('שם בנק/מקור', '')}: יתרת חוב {row.get('יתרת חוב', 0):,.0f} ₪ (פיגור: {row.get('יתרה שלא שולמה', 0):,.0f} ₪)")
            if not df_bank.empty:
                start_date = df_bank['Date'].min().strftime('%d/%m/%Y')
                end_date = df_bank['Date'].max().strftime('%d/%m/%Y')
                start_bal = df_bank.iloc[0]['Balance']
                end_bal = df_bank.iloc[-1]['Balance']
                financial_context_parts.append(f"\nמגמת יתרת בנק ({start_date} עד {end_date}): מ-{start_bal:,.0f} ₪ ל-{end_bal:,.0f} ₪.")

            system_prompt = (
                "אתה יועץ פיננסי מומחה לכלכלת המשפחה בישראל. תפקידך לספק ייעוץ פרקטי, ברור ואמפתי. "
                "ענה בעברית רהוטה. התבסס אך ורק על הנתונים המסוכמים הבאים של המשתמש. "
                "השתמש בסיווג (ירוק/צהוב/אדום) כבסיס להמלצותיך והרחב עליהן. אל תמציא נתונים. "
                "אם חסר מידע, ציין זאת. הדגש נקודות מרכזיות כמו יחס חוב-הכנסה והיתרה הפנויה. "
                "עזור למשתמש להבין את מצבו ולהתוות צעדים ראשונים אפשריים.\n\n"
                "--- סיכום נתוני המשתמש ---\n" + "\n".join(financial_context_parts) + "\n--- סוף נתונים ---"
            )

            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("שאל/י אותי כל שאלה..."):
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    try:
                        messages_for_api = [{"role": "system", "content": system_prompt}] + st.session_state.chat_messages
                        stream = client.chat.completions.create(model="gpt-4o-mini", messages=messages_for_api, stream=True)
                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                full_response += chunk.choices[0].delta.content
                                message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                    except APIError as e:
                        logging.error(f"OpenAI API Error: {e}", exc_info=True)
                        full_response = "מצטער, אירעה שגיאת תקשורת עם שירות הייעוץ. אנא נסה/י שוב מאוחר יותר."
                        message_placeholder.error(full_response)
                    except Exception as e:
                        logging.error(f"An unexpected error occurred during API call: {e}", exc_info=True)
                        full_response = "מצטער, אירעה שגיאה בלתי צפויה. אנא נסה/י שוב."
                        message_placeholder.error(full_response)
                st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
                st.rerun()
        else:
            st.warning("שירות הצ'אט אינו זמין. יש להגדיר מפתח API של OpenAI בסודות האפליקציה (secrets).")

        st.markdown('</div>', unsafe_allow_html=True)
