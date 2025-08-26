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
import fitz  # PyMuPDF
 # PyMuPDF, Hapoalim & Credit Report
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

# --- PDF Parsers (HAPOALIM, LEUMI, DISCOUNT, CREDIT REPORT) ---
# NOTE: The complex PDF parsing functions from the original script are included here.
# They are assumed to be correct and tailored to specific document formats.
# No logical changes were made to the parsers themselves.

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
                                logging.debug(f"Hapoalim: Found transaction - Date: {parsed_date}, Balance: {balance}, Line: {original_line.strip()}")
        except Exception as e:
            logging.error(f"Hapoalim: Error processing line {line_num+1} on page {page_num+1}: {e}", exc_info=True)
            continue

    doc.close()

    if not transactions:
        logging.warning(f"Hapoalim: No transactions found in {filename_for_logging}")
        return pd.DataFrame()

    df = pd.DataFrame(transactions)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce') # Ensure numeric, handle errors
    df = df.dropna(subset=['Date', 'Balance']) # Remove rows where date or balance parsing failed

    df = df.sort_values(by='Date').groupby('Date')['Balance'].last().reset_index()
    df = df.sort_values(by='Date').reset_index(drop=True) # Final sort

    logging.info(f"Hapoalim: Successfully extracted {len(df)} unique balance points from {filename_for_logging}")
    return df[['Date', 'Balance']]
# --- LEUMI PARSER ---
def clean_transaction_amount_leumi(text):
    """Cleans Leumi transaction amount, handles potential unicode zero-width space."""
    if text is None or pd.isna(text) or text == '': return None
    text = str(text).strip().replace('₪', '').replace(',', '')
    text = text.lstrip('\u200b')
    if text.count('.') > 1: # Handle cases like "1,234.56.78"
        parts = text.split('.')
        text = parts[0] + '.' + "".join(parts[1:])
    if '.' not in text: return None # Requires a decimal point
    try:
        val = float(text)
        if abs(val) > 100_000_000:
            logging.debug(f"Leumi: Transaction amount seems excessively large: {val} from '{text}'. Skipping.")
            return None
        return val
    except ValueError:
        logging.debug(f"Leumi: Could not convert amount '{text}' to float.");
        return None

def clean_number_leumi(text):
    """Specific cleaner for Leumi numbers (balances often). Uses general cleaner."""
    if text is None or pd.isna(text) or text == '': return None
    text = str(text).strip().replace('₪', '').replace(',', '')
    text = text.lstrip('\u200b')
    if text.count('.') > 1: # Handle cases like "1,234.56.78"
        parts = text.split('.')
        text = parts[0] + '.' + "".join(parts[1:])
    try:
        return float(text)
    except ValueError: return None

def parse_date_leumi(date_str):
    """Specific date parser for Leumi. Uses general parser."""
    return parse_date_general(date_str)

def normalize_text_leumi(text):
    """Normalizes Leumi text, including potential Hebrew reversal correction."""
    if text is None or pd.isna(text): return None
    text = str(text).replace('\r', ' ').replace('\n', ' ').replace('\u200b', '').strip()
    text = unicodedata.normalize('NFC', text)
    if any('\u0590' <= char <= '\u05EA' for char in text):
        words = text.split()
        reversed_text = ' '.join(words[::-1])
        return reversed_text
    return text

def parse_leumi_transaction_line_extracted_order_v2(line_text, previous_balance):
    """Attempts to parse a line assuming a specific column order from text extraction."""
    line = line_text.strip()
    if not line: return None


    pattern = re.compile(
        r"^([\-\u200b\d,\.]+)\s+"           # 1: Balance
        r"(\d{1,3}(?:,\d{3})*\.\d{2})?\s*"  # 2: Optional Amount
        r"(\S+)\s+"                         # 3: Reference (MANDATORY)
        r"(.*?)\s+"                         # 4: Description
        r"(\d{1,2}/\d{1,2}/\d{2,4})\s+"     # 5: First Date (e.g., Transaction Date)
        r"(\d{1,2}/\d{1,2}/\d{2,4})$"       # 6: Second Date (e.g., Value Date)
    )

    match = pattern.match(line)
    if not match:
        logging.debug(f"Leumi parse_line: No regex match for line: {line.strip()}")
        return None

    balance_str = match.group(1)
    amount_str = match.group(2)
    reference_str = match.group(3)
    description_raw = match.group(4)
    date_to_parse_str = match.group(5)

    parsed_date = parse_date_leumi(date_to_parse_str)
    if not parsed_date:
        logging.debug(f"Leumi parse_line: Failed to parse date '{date_to_parse_str}' from line: {line.strip()}")
        return None

    current_balance = clean_number_leumi(balance_str)
    if current_balance is None:
        logging.debug(f"Leumi parse_line: Failed to clean balance '{balance_str}' from line: {line.strip()}")
        return None

    amount = clean_transaction_amount_leumi(amount_str) # Can be None

    debit = None; credit = None
    if amount is not None and amount != 0 and previous_balance is not None:
        balance_diff = round(current_balance - previous_balance, 2)
        tolerance = 0.01
        if abs(balance_diff + amount) <= tolerance: debit = amount
        elif abs(balance_diff - amount) <= tolerance: credit = amount

    return {'Date': parsed_date, 'Balance': current_balance, 'Debit': debit, 'Credit': credit, 'Reference': reference_str, 'Description': normalize_text_leumi(description_raw)}
def extract_leumi_transactions_line_by_line(pdf_content_bytes, filename_for_logging="leumi_pdf"):
    """Extracts Date and Balance from Leumi PDF by processing lines."""
    transactions_data = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_content_bytes)) as pdf:
            previous_balance = None # Tracks the balance of the previously processed valid line
            first_transaction_processed = False # Flag to set the first previous_balance correctly
            logging.info(f"Starting Leumi PDF parsing for {filename_for_logging}")


            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2, layout=True)
                    if not text: continue

                    lines = text.splitlines()
                    for line_num, line_text in enumerate(lines):
                        normalized_line = normalize_text_leumi(line_text.strip())
                        if not normalized_line: continue

                        parsed_data = parse_leumi_transaction_line_extracted_order_v2(normalized_line, previous_balance)

                        if parsed_data and parsed_data['Balance'] is not None and parsed_data['Date'] is not None:
                            current_balance = parsed_data['Balance']
                            parsed_date = parsed_data['Date']

                            if not first_transaction_processed:
                                previous_balance = current_balance
                                first_transaction_processed = True

                            if parsed_data['Debit'] is not None or parsed_data['Credit'] is not None:
                                transactions_data.append({'Date': parsed_date, 'Balance': current_balance})
                                logging.debug(f"Leumi: Appended transaction - Date: {parsed_date}, Balance: {current_balance}, Line: {normalized_line.strip()}")
                                previous_balance = current_balance
                            else:
                                logging.debug(f"Leumi: Parsed line with balance but no Debit/Credit calculated, updating previous_balance: {normalized_line.strip()}")
                                previous_balance = current_balance
                        else:
                            logging.debug(f"Leumi: Line did not match transaction pattern or contained invalid data (skipped): {normalized_line.strip()}")
                            pass

                except Exception as e:
                     logging.error(f"Leumi: Error processing line {line_num+1} on page {page_num+1}: {e}", exc_info=True)
                     continue

    except Exception as e:
        logging.error(f"Leumi: FATAL ERROR processing PDF {filename_for_logging}: {e}", exc_info=True)
        return pd.DataFrame()

    if not transactions_data:
        logging.warning(f"Leumi: No transaction balances found in {filename_for_logging}")
        return pd.DataFrame()

    df = pd.DataFrame(transactions_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce') # Ensure numeric
    df = df.dropna(subset=['Date', 'Balance']) # Remove rows where date or balance parsing failed

    df = df.sort_values(by='Date').groupby('Date')['Balance'].last().reset_index()
    df = df.sort_values(by='Date').reset_index(drop=True) # Final sort

    logging.info(f"Leumi: Successfully extracted {len(df)} unique balance points from {filename_for_logging}")
    return df[['Date', 'Balance']]
# --- DISCOUNT PARSER ---
def parse_discont_transaction_line(line_text):
    """Attempts to parse a line from Discount assuming specific date/balance placement."""
    line = line_text.strip()
    if not line or len(line) < 20: return None

    balance_amount_pattern = re.compile(r"^([₪\-,\d]+\.\d{2})\s+([₪\-,\d]+\.\d{2})")
    balance_amount_match = balance_amount_pattern.search(line)

    if not balance_amount_match: return None

    balance_str = balance_amount_match.group(1)
    balance = clean_number_general(balance_str)

    if balance is None:
        logging.debug(f"Discount: Found dates but failed to clean balance: {balance_str} in line: {line.strip()}")
        return None

    date_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}/\d{1,2}/\d{2,4})$")
    date_match = date_pattern.search(line)
    if not date_match: return None

    date_str = date_match.group(1)
    parsed_date = parse_date_general(date_str)

    if not parsed_date:
        logging.debug(f"Discount: Failed to parse date '{date_str}' from line: {line.strip()}")
        return None

    lower_line = normalize_text_general(line).lower()
    if any(phrase in lower_line for phrase in ["יתרת סגירה", "יתרה נכון", "סך הכל", "סהכ", "עמוד", "הודעה זו כוללת"]):
         logging.debug(f"Discount: Skipping likely closing balance/summary/footer line: {line.strip()}")
         return None
    if any(header_part in lower_line for header_part in ["תאריך רישום", "תאריך ערך", "תיאור", "אסמכתא", "סכום", "יתרה"]):
         logging.debug(f"Discount: Skipping likely header line: {line.strip()}")
         return None

    logging.debug(f"Discount: Parsed transaction - Date: {parsed_date}, Balance: {balance}, Line: {line.strip()}")
    return {'Date': parsed_date, 'Balance': balance}
def extract_and_parse_discont_pdf(pdf_content_bytes, filename_for_logging="discount_pdf"):
    """Extracts Date and Balance from Discount PDF by processing lines."""
    transactions = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_content_bytes)) as pdf:
            logging.info(f"Starting Discount PDF parsing for {filename_for_logging}")
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2, layout=True)
                    if text:
                        lines = text.splitlines()
                        for line_num, line_text in enumerate(lines):
                            normalized_line = normalize_text_general(line_text)
                            parsed = parse_discont_transaction_line(normalized_line)
                            if parsed:
                                transactions.append(parsed)
                except Exception as e:
                    logging.error(f"Discount: Error processing page {page_num+1}: {e}", exc_info=True)
                    continue


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
    df = df.sort_values(by='Date').reset_index(drop=True)

    logging.info(f"Discount: Successfully extracted {len(df)} unique balance points from {filename_for_logging}")
    return df[['Date', 'Balance']]
# --- CREDIT REPORT PARSER ---
COLUMN_HEADER_WORDS_CR = {
"שם", "מקור", "מידע", "מדווח", "מזהה", "עסקה", "מספר", "עסקאות",
"גובה", "מסגרת", "מסגרות", "סכום", "הלוואות", "מקורי", "יתרת", "חוב",
"יתרה", "שלא", "שולמה", "במועד", "פרטי", "עסקה", "בנק", "אוצר",
"סוג", "מטבע", "מניין", "ימים", "ריבית", "ממוצעת"
}
BANK_KEYWORDS_CR = {"בנק", "בעמ", "אגוד", "דיסקונט", "לאומי", "הפועלים", "מזרחי",
"טפחות", "הבינלאומי", "מרכנתיל", "אוצר", "החייל", "ירושלים",
"איגוד", "מימון", "ישיר", "כרטיסי", "אשראי", "מקס", "פיננסים",
"כאל", "ישראכרט", "פועלים", "לאומי", "דיסקונט", "מזרחי", "טפחות", "בינלאומי", "מרכנתיל", "איגוד"}

def clean_credit_number(text):
    """Specific cleaner for credit report numbers, uses general."""
    return clean_number_general(text)

def process_entry_final_cr(entry_data, section, all_rows_list):
    """Processes a collected entry (bank name + numbers) into structured data."""
    if not entry_data or not entry_data.get('bank') or not entry_data.get('numbers'):
        logging.debug(f"CR: Skipping entry due to missing data: {entry_data}")
        return


    bank_name_raw = entry_data['bank']
    bank_name_cleaned = re.sub(r'\s*XX-[\w\d\-]+.*', '', bank_name_raw).strip()
    bank_name_cleaned = re.sub(r'\s+\d{1,3}(?:,\d{3})*$', '', bank_name_cleaned).strip()
    bank_name_cleaned = re.sub(r'\s+בע\"מ$', '', bank_name_cleaned, flags=re.IGNORECASE).strip()
    bank_name_cleaned = re.sub(r'\s+בנק$', '', bank_name_cleaned, flags=re.IGNORECASE).strip()
    bank_name_final = bank_name_cleaned if bank_name_cleaned else bank_name_raw

    is_likely_bank = any(kw in bank_name_final for kw in ["לאומי", "הפועלים", "דיסקונט", "מזרחי", "הבינלאומי", "מרכנתיל", "ירושלים", "איגוד", "טפחות", "אוצר"])
    if is_likely_bank and not bank_name_final.lower().endswith("בע\"מ"):
        bank_name_final += " בע\"מ"
    elif any(kw in bank_name_final for kw in ["מקס איט פיננסים", "מימון ישיר"]) and not bank_name_final.lower().endswith("בע\"מ"):
         bank_name_final += " בע\"מ"

    numbers_raw = entry_data['numbers']
    numbers = [clean_credit_number(n) for n in numbers_raw if clean_credit_number(n) is not None]

    num_count = len(numbers)
    limit_col, original_col, outstanding_col, unpaid_col = np.nan, np.nan, np.nan, np.nan

    if num_count >= 1:
        val1 = numbers[0] if num_count > 0 else np.nan
        val2 = numbers[1] if num_count > 1 else np.nan
        val3 = numbers[2] if num_count > 2 else np.nan
        val4 = numbers[3] if num_count > 3 else np.nan

        if section in ["עו\"ש", "מסגרת אשראי"]:
             if num_count >= 2:
                  limit_col = val1
                  outstanding_col = val2
                  unpaid_col = val3 if num_count > 2 else 0.0
             elif num_count == 1:
                  logging.debug(f"CR: Skipping עו\"ש/מסגרת entry for '{bank_name_final}' with only 1 number.")
                  return

        elif section in ["הלוואה", "משכנתה"]:
            if num_count >= 2:
                 if pd.notna(val1) and val1 == int(val1) and val1 > 0 and val1 < 600 and num_count >= 3:
                      original_col = val2
                      outstanding_col = val3
                      unpaid_col = val4 if num_count > 3 else 0.0
                 else:
                     original_col = val1
                     outstanding_col = val2 if num_count > 1 else np.nan
                     unpaid_col = val3 if num_count > 2 else 0.0
            elif num_count == 1:
                 outstanding_col = val1
                 original_col = np.nan
                 unpaid_col = 0.0
                 logging.debug(f"CR: Processing הלוואה/משכנתה entry for '{bank_name_final}' with only 1 number as Outstanding.")

        else: # Default case
            if num_count >= 2:
                 original_col = val1
                 outstanding_col = val2
                 unpaid_col = val3 if num_count > 2 else 0.0
            elif num_count == 1:
                 outstanding_col = val1
                 original_col = np.nan
                 unpaid_col = 0.0
            logging.debug(f"CR: Processing 'אחר' entry for '{bank_name_final}' with {num_count} numbers.")

        if pd.notna(outstanding_col) or pd.notna(limit_col):
             all_rows_list.append({
                 "סוג עסקה": section,
                 "שם בנק/מקור": bank_name_final,
                 "גובה מסגרת": limit_col,
                 "סכום מקורי": original_col,
                 "יתרת חוב": outstanding_col,
                 "יתרה שלא שולמה": unpaid_col
             })
             logging.debug(f"CR: Appended row: {all_rows_list[-1]}")
        else:
            logging.debug(f"CR: Skipping entry for '{bank_name_final}' as no outstanding or limit found after number parsing.")
def extract_credit_data_final_v13(pdf_content_bytes, filename_for_logging="credit_report_pdf"):
    """Extracts structured credit data from the report PDF."""
    extracted_rows = []
    try:
        with fitz.open(stream=pdf_content_bytes, filetype="pdf") as doc:
            current_section = None
            current_entry = None
            last_line_was_id = False
            potential_bank_continuation_candidate = False


            section_patterns = {
                "חשבון עובר ושב": "עו\"ש",
                "הלוואה": "הלוואה",
                "משכנתה": "משכנתה",
                "מסגרת אשראי מתחדשת": "מסגרת אשראי",
                "אחר": "אחר"
            }
            number_line_pattern = re.compile(r"^\s*(-?\d{1,3}(?:,\d{3})*\.?\d*)\s*$")
            id_line_pattern = re.compile(r"^XX-[\w\d\-]+.*$")

            logging.info(f"Starting Credit Report PDF parsing for {filename_for_logging}")

            for page_num, page in enumerate(doc):
                try:
                    lines = page.get_text("text", sort=True).splitlines()
                    logging.debug(f"Page {page_num + 1} has {len(lines)} lines.")

                    for line_num, line_text in enumerate(lines):
                        line = normalize_text_general(line_text)
                        if not line: potential_bank_continuation_candidate = False; continue

                        is_section_header = False
                        for header_keyword, section_name in section_patterns.items():
                            if header_keyword in line and len(line) < len(header_keyword) + 25 and line.count(' ') < 6:
                                if current_entry and not current_entry.get('processed', False):
                                    process_entry_final_cr(current_entry, current_section, extracted_rows)
                                current_section = section_name
                                current_entry = None
                                last_line_was_id = False
                                potential_bank_continuation_candidate = False
                                is_section_header = True
                                logging.debug(f"CR: Detected section header: {line} -> {current_section}")
                                break
                        if is_section_header: continue

                        if line.startswith("סה\"כ") or line.startswith("הודעה זו כוללת") or "עמוד" in line:
                            if current_entry and not current_entry.get('processed', False):
                                process_entry_final_cr(current_entry, current_section, extracted_rows)
                            current_entry = None
                            last_line_was_id = False
                            potential_bank_continuation_candidate = False
                            logging.debug(f"CR: Detected summary/footer line: {line}")
                            continue

                        number_match = number_line_pattern.match(line)
                        is_id_line = id_line_pattern.match(line)
                        is_noise_line = any(word in line.split() for word in COLUMN_HEADER_WORDS_CR) or line in [':', '.', '-', '—'] or (len(line.replace(' ','')) < 3 and not line.replace(' ','').isdigit()) or re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", line)

                        if number_match:
                            if current_entry:
                                try:
                                    number_str = number_match.group(1)
                                    number = clean_credit_number(number_str)
                                    if number is not None:
                                        num_list = current_entry.get('numbers', [])
                                        if last_line_was_id:
                                            if current_entry and not current_entry.get('processed', False):
                                                 process_entry_final_cr(current_entry, current_section, extracted_rows)
                                            current_entry = {'bank': current_entry['bank'], 'numbers': [number], 'processed': False}
                                            logging.debug(f"CR: Detected number after ID line, starting new entry for bank '{current_entry['bank']}' with first number: {number}")
                                        else:
                                             if len(num_list) < 5:
                                                 current_entry['numbers'].append(number)
                                                 logging.debug(f"CR: Added number {number} to current entry for bank '{current_entry.get('bank', 'N/A')}'. Numbers: {current_entry['numbers']}")
                                             else:
                                                 logging.debug(f"CR: Skipping extra number {number} for bank '{current_entry.get('bank', 'N/A')}'. Max numbers reached.")

                                except Exception as e:
                                    logging.error(f"CR: Error processing number line '{line.strip()}': {e}", exc_info=True)

                            last_line_was_id = False
                            potential_bank_continuation_candidate = False
                            continue

                        elif is_id_line:
                            last_line_was_id = True
                            potential_bank_continuation_candidate = False
                            logging.debug(f"CR: Detected ID line: {line}")
                            continue

                        elif is_noise_line:
                            last_line_was_id = False
                            potential_bank_continuation_candidate = False
                            logging.debug(f"CR: Skipping likely noise line: {line}")
                            continue

                        else:
                            cleaned_line = re.sub(r'\s*XX-[\w\d\-]+.*|\s+\d+$', '', line).strip()
                            common_continuations = ["לישראל", "בע\"מ", "ומשכנתאות", "נדל\"ן", "דיסקונט", "הראשון", "פיננסים", "איגוד", "אשראי", "חברה", "למימון", "שירותים"]

                            seems_like_continuation_text = any(cleaned_line.startswith(cont) for cont in common_continuations) or \
                                                           (len(cleaned_line) > 3 and ' ' in cleaned_line and not any(char.isdigit() for char in cleaned_line))

                            if potential_bank_continuation_candidate and current_entry and seems_like_continuation_text:
                                current_entry['bank'] = (current_entry['bank'] + " " + cleaned_line).replace(" בע\"מ בע\"מ", " בע\"מ").strip()
                                logging.debug(f"CR: Appended continuation '{cleaned_line}' to bank name. New bank name: '{current_entry['bank']}'")
                                potential_bank_continuation_candidate = True
                            elif len(cleaned_line) > 3 and any(kw in cleaned_line for kw in BANK_KEYWORDS_CR) and not any(char.isdigit() for char in cleaned_line):
                                 if current_entry and not current_entry.get('processed', False):
                                      process_entry_final_cr(current_entry, current_section, extracted_rows)
                                 current_entry = {'bank': cleaned_line, 'numbers': [], 'processed': False}
                                 potential_bank_continuation_candidate = True
                                 logging.debug(f"CR: Started new entry with bank name: '{cleaned_line}'")
                            else:
                                  if current_entry and current_entry.get('numbers') and not current_entry.get('processed', False):
                                       process_entry_final_cr(current_entry, current_section, extracted_rows)
                                       current_entry['processed'] = True
                                  potential_bank_continuation_candidate = False

                            last_line_was_id = False

                except Exception as e:
                    logging.error(f"CR: Error processing line {line_num+1} on page {page_num+1}: {e}", exc_info=True)
                    continue

            if current_entry and not current_entry.get('processed', False):
                process_entry_final_cr(current_entry, current_section, extracted_rows)

    except Exception as e:
        logging.error(f"CreditReport: FATAL ERROR processing {filename_for_logging}: {e}", exc_info=True)
        return pd.DataFrame()

    if not extracted_rows:
        logging.warning(f"CreditReport: No structured entries found in {filename_for_logging}")
        return pd.DataFrame()

    df = pd.DataFrame(extracted_rows)

    final_cols = ["סוג עסקה", "שם בנק/מקור", "גובה מסגרת", "סכום מקורי", "יתרת חוב", "יתרה שלא שולמה"]
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[final_cols]

    for col in ["גובה מסגרת", "סכום מקורי", "יתרת חוב", "יתרה שלא שולמה"]:
        if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce')
             if col == "יתרה שלא שולמה":
                  df[col] = df[col].fillna(0)

    df = df.dropna(subset=['גובה מסגרת', 'סכום מקורי', 'יתרת חוב', 'יתרה שלא שולמה'], how='all').reset_index(drop=True)

    logging.info(f"CreditReport: Successfully extracted {len(df)} entries from {filename_for_logging}")

    return df

# --- Initialize Session State ---
# This function ensures that all necessary variables are initialized
# at the start of the session to prevent errors.
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
    # Clear all keys, then re-initialize
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
                "בחר את סוג דוח הבנק:",
                bank_type_options,
                key="bank_type_selector_main"
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
            error_processing = False
            with st.spinner("מעבד קבצים... אנא המתן/י..."):
                # Process Bank File
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
                            st.warning(f"לא חולצו נתונים מדוח הבנק ({st.session_state.bank_type_selected}). ייתכן שהפורמט אינו נתמך.")
                            error_processing = True
                    except Exception as e:
                        logging.error(f"Error processing bank file {uploaded_bank_file.name}: {e}", exc_info=True)
                        st.error(f"שגיאה בעיבוד דוח הבנק: {e}")
                        error_processing = True

                # Process Credit File
                if uploaded_credit_file:
                    try:
                        st.session_state.df_credit_uploaded = extract_credit_data_final_v13(uploaded_credit_file.getvalue(), uploaded_credit_file.name)
                        if st.session_state.df_credit_uploaded.empty:
                            st.warning("לא חולצו נתונים מדוח האשראי.")
                            error_processing = True
                        elif 'יתרת חוב' in st.session_state.df_credit_uploaded.columns:
                            total_debt = st.session_state.df_credit_uploaded['יתרת חוב'].fillna(0).sum()
                            st.session_state.total_debt_from_credit_report = total_debt
                    except Exception as e:
                        logging.error(f"Error processing credit file {uploaded_credit_file.name}: {e}", exc_info=True)
                        st.error(f"שגיאה בעיבוד דוח האשראי: {e}")
                        error_processing = True

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

    # Progress bar
    total_stages = 4
    current_stage = st.session_state.questionnaire_stage
    # For stage 100, show progress as complete
    progress_value = 1.0 if current_stage == 100 or current_stage == -1 else (current_stage + 1) / total_stages
    st.progress(progress_value, text=f"שלב {current_stage+1 if current_stage < 100 else total_stages} מתוך {total_stages}")

    q_stage = st.session_state.questionnaire_stage

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if q_stage == 0:
            st.subheader("חלק א': שאלות פתיחה")
            st.text_area("1. האם קרה אירוע חריג שבעקבותיו פניתם לייעוץ?", value=st.session_state.answers.get('q1_unusual_event', ''), key="q_s0_q1")
            st.text_area("2. האם בדקתם מקורות מימון או פתרונות אחרים?", value=st.session_state.answers.get('q2_other_funding', ''), key="q_s0_q2")
            st.radio("3. האם קיימות הלוואות נוספות (לא משכנתא)?", ("כן", "לא"), index=("כן", "לא").index(st.session_state.answers.get('q3_existing_loans_bool_radio', 'לא')), key="q3_existing_loans_bool_radio")
            if st.session_state.q3_existing_loans_bool_radio == "כן":
                st.number_input("מה גובה ההחזר החודשי הכולל עליהן?", min_value=0.0, value=float(st.session_state.answers.get('q3_loan_repayment_amount', 0.0)), step=100.0, key="q3_loan_repayment_amount")
            else: st.session_state.answers['q3_loan_repayment_amount'] = 0.0
            st.radio("4. האם אתם מאוזנים כלכלית (הכנסות מכסות הוצאות)?", ("כן", "בערך", "לא"), index=("כן", "בערך", "לא").index(st.session_state.answers.get('q4_financially_balanced_bool_radio', 'כן')), key="q4_financially_balanced_bool_radio")
            st.text_area("האם מצבכם הכלכלי צפוי להשתנות משמעותית בשנה הקרובה?", value=st.session_state.answers.get('q4_situation_change_next_year', ''), key="q4_situation_change_next_year")

        elif q_stage == 1:
            st.subheader("חלק ב': הכנסות (נטו חודשי)")
            st.number_input("הכנסתך (נטו):", min_value=0.0, value=float(st.session_state.answers.get('income_employee', 0.0)), step=100.0, key="income_employee")
            st.number_input("הכנסת בן/בת הזוג (נטו):", min_value=0.0, value=float(st.session_state.answers.get('income_partner', 0.0)), step=100.0, key="income_partner")
            st.number_input("הכנסות נוספות (קצבאות, שכירות וכו'):", min_value=0.0, value=float(st.session_state.answers.get('income_other', 0.0)), step=100.0, key="income_other")
            total_net_income = st.session_state.income_employee + st.session_state.income_partner + st.session_state.income_other
            st.session_state.answers['total_net_income'] = total_net_income
            st.metric("💰 סך הכנסות נטו (חודשי):", f"{total_net_income:,.0f} ₪")

        elif q_stage == 2:
            st.subheader("חלק ג': הוצאות קבועות חודשיות")
            st.number_input("שכירות / החזר משכנתא:", min_value=0.0, value=float(st.session_state.answers.get('expense_rent_mortgage', 0.0)), step=100.0, key="expense_rent_mortgage")
            default_debt_repayment = float(st.session_state.answers.get('q3_loan_repayment_amount', 0.0))
            st.number_input("החזרי הלוואות נוספות:", min_value=0.0, value=float(st.session_state.answers.get('expense_debt_repayments', default_debt_repayment)), step=100.0, key="expense_debt_repayments")
            st.number_input("מזונות / הוצאות קבועות גדולות אחרות:", min_value=0.0, value=float(st.session_state.answers.get('expense_alimony_other', 0.0)), step=100.0, key="expense_alimony_other")
            total_fixed_expenses = st.session_state.expense_rent_mortgage + st.session_state.expense_debt_repayments + st.session_state.expense_alimony_other
            st.session_state.answers['total_fixed_expenses'] = total_fixed_expenses
            total_net_income = float(st.session_state.answers.get('total_net_income', 0.0))
            monthly_balance = total_net_income - total_fixed_expenses
            st.metric("💸 סך הוצאות קבועות:", f"{total_fixed_expenses:,.0f} ₪")
            st.metric("📊 יתרה פנויה חודשית:", f"{monthly_balance:,.0f} ₪", delta_color="inverse")
            if monthly_balance < 0: st.warning("שימו לב: ההוצאות הקבועות גבוהות מההכנסות.")

        elif q_stage == 3:
            st.subheader("חלק ד': חובות ופיגורים")
            default_total_debt = st.session_state.total_debt_from_credit_report if st.session_state.total_debt_from_credit_report is not None else 0.0
            if st.session_state.total_debt_from_credit_report is not None:
                st.info(f"סך יתרת החוב שחושבה מדוח האשראי הוא: {default_total_debt:,.0f} ₪. ניתן לעדכן את הסכום אם קיימים חובות נוספים.")
            else:
                 st.info("אנא הזן/י את סך כל החובות הקיימים (למעט משכנתא).")
            st.number_input("מה היקף החובות הכולל (למעט משכנתא)?", min_value=0.0, value=float(st.session_state.answers.get('total_debt_amount', default_total_debt)), step=100.0, key="total_debt_amount")
            st.radio("האם קיימים פיגורים משמעותיים בתשלומים או הליכי גבייה?",("כן", "לא"), index=("כן", "לא").index(st.session_state.answers.get('arrears_collection_proceedings_radio', 'לא')), key="arrears_collection_proceedings_radio")

        elif q_stage == 100:
            st.subheader("שאלות הבהרה נוספות")
            st.warning(f"תוצאה ראשונית: יחס החוב להכנסה שלך הוא {st.session_state.answers.get('debt_to_income_ratio', 0.0):.2f}. ({st.session_state.classification_details.get('description')})")
            if st.session_state.answers.get('arrears_collection_proceedings_radio', 'לא') == 'כן':
                 st.error("זוהו הליכי גבייה. מצב זה מסווג אוטומטית כ'אדום'.")
            else:
                total_debt = float(st.session_state.answers.get('total_debt_amount', 0.0))
                fifty_percent_debt = total_debt * 0.5
                st.radio(f"האם תוכל/י לגייס סכום של כ-50% מהחוב ({fifty_percent_debt:,.0f} ₪) ממקורות תמיכה (משפחה, חברים, מימוש נכסים) בזמן סביר?",("כן", "לא"), index=("כן", "לא").index(st.session_state.answers.get('can_raise_50_percent_radio', 'לא')), key="can_raise_50_percent_radio")

        st.divider()

        # Navigation Buttons
        cols = st.columns([1, 5, 1])
        if q_stage > 0:
            if cols[0].button("⬅️ הקודם", key=f"q_s{q_stage}_prev", use_container_width=True):
                if q_stage == 100: st.session_state.questionnaire_stage = 3
                else: st.session_state.questionnaire_stage -= 1
                st.rerun()

        if q_stage < 3:
            if cols[2].button("הבא ➡️", key=f"q_s{q_stage}_next", use_container_width=True):
                # Before moving, save current answers from widgets to session state
                for key, val in st.session_state.items():
                    if key.startswith('q_s') or key in ['income_employee', 'income_partner', 'income_other', 'expense_rent_mortgage', 'expense_debt_repayments', 'expense_alimony_other', 'total_debt_amount', 'arrears_collection_proceedings_radio', 'q3_existing_loans_bool_radio', 'q4_financially_balanced_bool_radio']:
                        st.session_state.answers[key] = val
                st.session_state.questionnaire_stage += 1
                st.rerun()

        elif q_stage == 3:
            if cols[2].button("✅ סיום וקבלת סיכום", key="q_s3_next_finish", use_container_width=True):
                # Save final answers
                for key in ['total_debt_amount', 'arrears_collection_proceedings_radio']: st.session_state.answers[key] = st.session_state[key]

                # Classification Logic
                total_debt = float(st.session_state.answers.get('total_debt_amount', 0.0))
                net_income = float(st.session_state.answers.get('total_net_income', 0.0))
                annual_income = net_income * 12
                st.session_state.answers['annual_income'] = annual_income
                ratio = (total_debt / annual_income) if annual_income > 0 else float('inf')
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
                    st.session_state.questionnaire_stage = 100 # Go to intermediate stage
                else: # ratio > 2
                    st.session_state.classification_details = {'classification': "אדום", 'description': "יחס חוב להכנסה גבוה משנתיים הכנסה.", 'color': "red"}
                    st.session_state.app_stage = "summary"

                st.rerun()

        elif q_stage == 100:
             if cols[2].button("המשך לסיכום ➡️", key="q_s100_to_summary", use_container_width=True):
                st.session_state.answers['can_raise_50_percent_radio'] = st.session_state.can_raise_50_percent_radio
                arrears = st.session_state.answers.get('arrears_collection_proceedings_radio', 'לא') == 'כן'
                can_raise_funds = st.session_state.answers.get('can_raise_50_percent_radio', 'לא') == 'כן'

                if arrears:
                    st.session_state.classification_details.update({'classification': "אדום", 'description': "יחס חוב להכנסה בינוני אך קיימים הליכי גבייה.", 'color': "red"})
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
        if not st.session_state.df_credit_uploaded.empty and 'יתרת חוב' in st.session_state.df_credit_uploaded.columns:
            df_credit = st.session_state.df_credit_uploaded.copy()
            df_credit['יתרת חוב'] = pd.to_numeric(df_credit['יתרת חוב'], errors='coerce').fillna(0)
            debt_summary = df_credit.groupby("סוג עסקה")["יתרת חוב"].sum().reset_index()
            debt_summary = debt_summary[debt_summary['יתרת חוב'] > 0]
            if not debt_summary.empty:
                chart_pie = alt.Chart(debt_summary).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="יתרת חוב", type="quantitative"),
                    color=alt.Color(field="סוג עסקה", type="nominal"),
                    tooltip=["סוג עסקה", "יתרת חוב"]
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
                x=alt.X('קטגוריה', sort=None),
                y='סכום',
                color='קטגוריה',
                tooltip=['קטגוריה', 'סכום']
            ).properties(title="השוואת חובות להכנסה שנתית")
            st.altair_chart(chart_bar, use_container_width=True)
        else:
            st.info("אין נתוני חוב או הכנסה להצגת השוואה.")
        st.markdown('</div>', unsafe_allow_html=True)

# Bank Balance Trend (Altair Line Chart)
if not st.session_state.df_bank_uploaded.empty:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        df_bank = st.session_state.df_bank_uploaded.dropna(subset=['Date', 'Balance']).sort_values('Date')
        if not df_bank.empty:
            chart_line = alt.Chart(df_bank).mark_line(point=True).encode(
                x=alt.X('Date:T', title="תאריך"),
                y=alt.Y('Balance:Q', title="יתרה"),
                tooltip=['Date', 'Balance']
            ).properties(title="מגמת יתרת חשבון הבנק")
            st.altair_chart(chart_line, use_container_width=True)
        else:
            st.info("אין נתוני יתרה תקינים בדוח הבנק להצגה.")
        st.markdown('</div>', unsafe_allow_html=True)


    # --- Raw Data Expander ---
    with st.expander("הצג נתונים מפורטים שחולצו מהדוחות"):
        if not st.session_state.df_credit_uploaded.empty:
            st.write("נתוני דוח אשראי:")
            st.dataframe(st.session_state.df_credit_uploaded.style.format(precision=0, thousands=","), use_container_width=True)
        if not st.session_state.df_bank_uploaded.empty:
            st.write(f"נתוני דוח בנק ({st.session_state.bank_type_selected}):")
            st.dataframe(st.session_state.df_bank_uploaded.style.format({"Balance": '{:,.2f}'}), use_container_width=True)

    # --- Chatbot Interface ---
    st.divider()
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
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
            # Add credit report details if available
            if not st.session_state.df_credit_uploaded.empty:
                financial_context_parts.append("\nפירוט חובות מדוח אשראי:")
                for _, row in st.session_state.df_credit_uploaded.head(10).iterrows():
                    financial_context_parts.append(f"  - {row.get('סוג עסקה', '')} ב{row.get('שם בנק/מקור', '')}: יתרת חוב {row.get('יתרת חוב', 0):,.0f} ₪ (פיגור: {row.get('יתרה שלא שולמה', 0):,.0f} ₪)")
            # Add bank trend info if available
            if not st.session_state.df_bank_uploaded.empty and not df_bank.empty:
                start_date = df_bank['Date'].min().strftime('%d/%m/%Y')
                end_date = df_bank['Date'].max().strftime('%d/%m/%Y')
                start_bal = df_bank.iloc[0]['Balance']
                end_bal = df_bank.iloc[-1]['Balance']
                financial_context_parts.append(f"\nמגמת יתרת בנק ({start_date} עד {end_date}): מ-{start_bal:,.0f} ₪ ל-{end_bal:,.0f} ₪.")

            # System Prompt
            system_prompt = (
                "אתה יועץ פיננסי מומחה לכלכלת המשפחה בישראל. תפקידך לספק ייעוץ פרקטי, ברור ואמפתי. "
                "ענה בעברית רהוטה. התבסס אך ורק על הנתונים המסוכמים הבאים של המשתמש. "
                "השתמש בסיווג (ירוק/צהוב/אדום) כבסיס להמלצותיך והרחב עליהן. אל תמציא נתונים. "
                "אם חסר מידע, ציין זאת. הדגש נקודות מרכזיות כמו יחס חוב-הכנסה והיתרה הפנויה. "
                "עזור למשתמש להבין את מצבו ולהתוות צעדים ראשונים אפשריים.\n\n"
                "--- סיכום נתוני המשתמש ---\n" + "\n".join(financial_context_parts) + "\n--- סוף נתונים ---"
            )

            # Display chat history
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Handle new input
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

