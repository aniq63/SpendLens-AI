import gradio as gr
import sqlite3
import hashlib
import datetime
from PIL import Image
import pytesseract
import os
import json
import calendar
from typing import Dict, List, Tuple
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Set up Groq API key
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize LLM
llm = ChatGroq(model_name="llama-3.1-8b-instant")

# Pydantic models for structured output
class ReceiptItem(BaseModel):
    item_name: str = Field(description="Name of the item")
    quantity: int = Field(description="Quantity of the item")
    unit_price: float = Field(description="Unit price of the item")
    total_price: float = Field(description="Total price for this item")
    category: str = Field(description="Category of the item (e.g., Food, Beverages, Electronics, etc.)")

class ReceiptAnalysis(BaseModel):
    items: List[ReceiptItem] = Field(description="List of items from the receipt")
    subtotal: float = Field(description="Subtotal amount")
    tax_amount: float = Field(description="Tax amount paid")
    total_amount: float = Field(description="Total amount paid")
    store_name: str = Field(description="Name of the store", default="Unknown")
    date: str = Field(description="Date of purchase", default="Unknown")

# Database setup
def init_database():
    conn = sqlite3.connect('spendlens.db')
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Receipts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            store_name TEXT,
            total_amount REAL NOT NULL,
            tax_amount REAL NOT NULL,
            subtotal REAL NOT NULL,
            purchase_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Receipt items table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS receipt_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            receipt_id INTEGER NOT NULL,
            item_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            total_price REAL NOT NULL,
            category TEXT NOT NULL,
            FOREIGN KEY (receipt_id) REFERENCES receipts (id)
        )
    ''')

    conn.commit()
    conn.close()

# User authentication functions
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username: str, password: str) -> Tuple[bool, str]:
    if not username or not password:
        return False, "Username and password cannot be empty"

    if len(password) < 6:
        return False, "Password must be at least 6 characters long"

    conn = sqlite3.connect('spendlens.db')
    cursor = conn.cursor()

    try:
        cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                      (username, hash_password(password)))
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    finally:
        conn.close()

def login_user(username: str, password: str) -> Tuple[bool, int, str]:
    if not username or not password:
        return False, -1, "Username and password cannot be empty"

    conn = sqlite3.connect('spendlens.db')
    cursor = conn.cursor()

    cursor.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()

    if user and user[1] == hash_password(password):
        return True, user[0], "Login successful!"
    else:
        return False, -1, "Invalid username or password"

# OCR and LLM processing
def extract_text_from_receipt(image):
    """Extract text from receipt image using OCR"""
    try:
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text
    except Exception as e:
        return f"Error in OCR processing: {str(e)}"

def analyze_receipt_with_llm(receipt_text: str) -> ReceiptAnalysis:
    """Analyze receipt text using LLM and return structured data"""

    # Create parser
    parser = PydanticOutputParser(pydantic_object=ReceiptAnalysis)

    # System prompt template
    system_prompt_template = """
    You're an expert finance assistant. Categorize the items from the receipt and extract all relevant information.

    For each item, identify:
    - Item name
    - Quantity
    - Unit price
    - Total price for that item
    - Category (Food & Beverages, Electronics, Clothing, Healthcare, Transportation, Entertainment, etc.)

    Also extract:
    - Store name
    - Purchase date
    - Subtotal
    - Tax amount
    - Total amount paid

    Receipt Text:
    {receipt_text}

    {format_instructions}
    """

    # Create prompt
    format_instructions = parser.get_format_instructions()
    prompt = PromptTemplate(
        template=system_prompt_template,
        input_variables=["receipt_text"],
        partial_variables={"format_instructions": format_instructions}
    )

    # Create chain and process
    chain = prompt | llm | parser
    try:
        output = chain.invoke({"receipt_text": receipt_text})
        return output
    except Exception as e:
        # Return default structure if parsing fails
        return ReceiptAnalysis(
            items=[],
            subtotal=0.0,
            tax_amount=0.0,
            total_amount=0.0,
            store_name="Error in processing",
            date=str(datetime.date.today())
        )

def save_receipt_to_database(user_id: int, analysis: ReceiptAnalysis) -> int:
    """Save receipt analysis to database"""
    conn = sqlite3.connect('spendlens.db')
    cursor = conn.cursor()

    # Insert receipt
    cursor.execute('''
        INSERT INTO receipts (user_id, store_name, total_amount, tax_amount, subtotal, purchase_date)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, analysis.store_name, analysis.total_amount, analysis.tax_amount,
          analysis.subtotal, analysis.date))

    receipt_id = cursor.lastrowid

    # Insert receipt items
    for item in analysis.items:
        cursor.execute('''
            INSERT INTO receipt_items (receipt_id, item_name, quantity, unit_price, total_price, category)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (receipt_id, item.item_name, item.quantity, item.unit_price,
              item.total_price, item.category))

    conn.commit()
    conn.close()
    return receipt_id

def get_user_stats(user_id: int) -> Dict:
    """Get user statistics for dashboard"""
    conn = sqlite3.connect('spendlens.db')
    cursor = conn.cursor()

    # Get current month stats
    current_month = datetime.datetime.now().month
    current_year = datetime.datetime.now().year

    cursor.execute('''
        SELECT
            SUM(total_amount) as total_spent,
            SUM(tax_amount) as total_tax,
            COUNT(*) as receipt_count
        FROM receipts
        WHERE user_id = ? AND strftime('%Y', created_at) = ? AND strftime('%m', created_at) = ?
    ''', (user_id, str(current_year), f"{current_month:02d}"))

    current_stats = cursor.fetchone()

    # Get last month stats for comparison
    last_month = current_month - 1 if current_month > 1 else 12
    last_year = current_year if current_month > 1 else current_year - 1

    cursor.execute('''
        SELECT
            SUM(total_amount) as total_spent,
            SUM(tax_amount) as total_tax,
            COUNT(*) as receipt_count
        FROM receipts
        WHERE user_id = ? AND strftime('%Y', created_at) = ? AND strftime('%m', created_at) = ?
    ''', (user_id, str(last_year), f"{last_month:02d}"))

    last_stats = cursor.fetchone()

    # Get recent receipts
    cursor.execute('''
        SELECT store_name, total_amount, created_at, id
        FROM receipts
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 5
    ''', (user_id,))

    recent_receipts = cursor.fetchall()

    conn.close()

    # Calculate percentage changes
    current_spent = current_stats[0] if current_stats[0] else 0
    current_tax = current_stats[1] if current_stats[1] else 0
    current_receipts = current_stats[2] if current_stats[2] else 0

    last_spent = last_stats[0] if last_stats and last_stats[0] else 0
    last_tax = last_stats[1] if last_stats and last_stats[1] else 0

    spent_change = ((current_spent - last_spent) / last_spent * 100) if last_spent > 0 else 0
    tax_change = ((current_tax - last_tax) / last_tax * 100) if last_tax > 0 else 0

    return {
        'total_spent': current_spent,
        'total_tax': current_tax,
        'receipt_count': current_receipts,
        'spent_change': spent_change,
        'tax_change': tax_change,
        'recent_receipts': recent_receipts
    }

def get_monthly_spending(user_id: int, year: int, month: int) -> Dict:
    """Get spending summary for a specific month"""
    conn = sqlite3.connect('spendlens.db')
    cursor = conn.cursor()

    # Get receipts for the month
    cursor.execute('''
        SELECT r.*, COUNT(ri.id) as item_count
        FROM receipts r
        LEFT JOIN receipt_items ri ON r.id = ri.receipt_id
        WHERE r.user_id = ? AND strftime('%Y', r.created_at) = ? AND strftime('%m', r.created_at) = ?
        GROUP BY r.id
        ORDER BY r.created_at DESC
    ''', (user_id, str(year), f"{month:02d}"))

    receipts = cursor.fetchall()

    # Get category-wise spending
    cursor.execute('''
        SELECT ri.category, SUM(ri.total_price) as total_spent, COUNT(ri.id) as item_count
        FROM receipt_items ri
        JOIN receipts r ON ri.receipt_id = r.id
        WHERE r.user_id = ? AND strftime('%Y', r.created_at) = ? AND strftime('%m', r.created_at) = ?
        GROUP BY ri.category
        ORDER BY total_spent DESC
    ''', (user_id, str(year), f"{month:02d}"))

    categories = cursor.fetchall()

    # Calculate totals
    total_spent = sum([r[3] for r in receipts])
    total_tax = sum([r[4] for r in receipts])

    conn.close()

    return {
        'receipts': receipts,
        'categories': categories,
        'total_spent': total_spent,
        'total_tax': total_tax,
        'receipt_count': len(receipts)
    }

# Gradio Interface Functions
def process_receipt(image, user_session):
    """Main function to process uploaded receipt"""
    if user_session is None or not user_session.get('logged_in', False):
        return "❌ Please log in first!", None, ""

    if image is None:
        return "❌ Please upload a receipt image!", None, ""

    try:
        # Step 1: Extract text using OCR
        extracted_text = extract_text_from_receipt(image)

        if "Error" in extracted_text:
            return extracted_text, None, ""

        # Step 2: Analyze with LLM
        analysis = analyze_receipt_with_llm(extracted_text)

        # Step 3: Save to database
        receipt_id = save_receipt_to_database(user_session['user_id'], analysis)

        # Step 4: Create table data for items
        table_data = []
        for item in analysis.items:
            table_data.append([
                item.item_name,
                item.category,
                item.quantity,
                f"${item.unit_price:.2f}",
                f"${item.total_price:.2f}"
            ])

        # Step 5: Format summary
        summary = f"""
## 🧾 Receipt Analysis Complete!

**🏪 Store:** {analysis.store_name}
**📅 Date:** {analysis.date}
**🆔 Receipt ID:** #{receipt_id}

### 💰 Financial Summary:
- **Subtotal:** ${analysis.subtotal:.2f}
- **Tax:** ${analysis.tax_amount:.2f}
- **Total Amount:** ${analysis.total_amount:.2f}
"""

        return "✅ Receipt processed successfully!", table_data, summary

    except Exception as e:
        return f"❌ Error processing receipt: {str(e)}", None, ""

def show_dashboard(user_session):
    """Show user dashboard with stats"""
    if user_session is None or not user_session.get('logged_in', False):
        return "Please log in first!", None

    try:
        stats = get_user_stats(user_session['user_id'])

        # Create dashboard summary
        dashboard_html = f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; text-align: center;">
                <h3 style="margin: 0; font-size: 14px; opacity: 0.9;">Total Spent</h3>
                <h2 style="margin: 10px 0; font-size: 28px;">${stats['total_spent']:.2f}</h2>
                <p style="margin: 0; font-size: 12px; opacity: 0.8;">
                    {'+' if stats['spent_change'] >= 0 else ''}{stats['spent_change']:.1f}% from last month
                </p>
            </div>
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 12px; text-align: center;">
                <h3 style="margin: 0; font-size: 14px; opacity: 0.9;">Tax Paid</h3>
                <h2 style="margin: 10px 0; font-size: 28px;">${stats['total_tax']:.2f}</h2>
                <p style="margin: 0; font-size: 12px; opacity: 0.8;">
                    {'+' if stats['tax_change'] >= 0 else ''}{stats['tax_change']:.1f}% from last month
                </p>
            </div>
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 20px; border-radius: 12px; text-align: center;">
                <h3 style="margin: 0; font-size: 14px; opacity: 0.9;">Receipts Processed</h3>
                <h2 style="margin: 10px 0; font-size: 28px;">{stats['receipt_count']}</h2>
                <p style="margin: 0; font-size: 12px; opacity: 0.8;">
                    This month
                </p>
            </div>
        </div>
        """

        # Recent receipts table
        recent_table = []
        for receipt in stats['recent_receipts']:
            date_obj = datetime.datetime.strptime(receipt[2], '%Y-%m-%d %H:%M:%S')
            formatted_date = date_obj.strftime('%B %d, %Y')
            recent_table.append([
                receipt[0],  # store_name
                formatted_date,  # formatted date
                f"${receipt[1]:.2f}"  # total_amount
            ])

        return dashboard_html, recent_table

    except Exception as e:
        return f"Error loading dashboard: {str(e)}", None

def show_monthly_calendar(year, month, user_session):
    """Show monthly spending calendar"""
    if user_session is None or not user_session.get('logged_in', False):
        return "Please log in first!"

    try:
        spending_data = get_monthly_spending(user_session['user_id'], year, month)
        month_name = calendar.month_name[month]

        output = f"""
# 📊 Monthly Report - {month_name} {year}

## Summary:
- **💰 Total Spent:** ${spending_data['total_spent']:.2f}
- **🧾 Tax Paid:** ${spending_data['total_tax']:.2f}
- **📋 Receipts:** {spending_data['receipt_count']}

## 📊 Category Breakdown:
"""

        for category, total, count in spending_data['categories']:
            output += f"- **{category}:** ${total:.2f} ({count} items)\n"

        return output

    except Exception as e:
        return f"Error loading monthly data: {str(e)}"

def handle_login(username, password):
    """Handle user login"""
    success, user_id, message = login_user(username, password)
    if success:
        return message, {"logged_in": True, "user_id": user_id, "username": username}, gr.update(visible=False), gr.update(visible=True)
    else:
        return message, None, gr.update(visible=True), gr.update(visible=False)

def handle_signup(username, password, confirm_password):
    """Handle user registration"""
    if password != confirm_password:
        return "❌ Passwords do not match!", None, gr.update(visible=True), gr.update(visible=False)

    success, message = register_user(username, password)
    if success:
        # Auto login after successful registration
        _, user_id, _ = login_user(username, password)
        return "✅ " + message + " You are now logged in!", {"logged_in": True, "user_id": user_id, "username": username}, gr.update(visible=False), gr.update(visible=True)
    else:
        return "❌ " + message, None, gr.update(visible=True), gr.update(visible=False)

def logout_user():
    """Handle user logout"""
    return None, gr.update(visible=True), gr.update(visible=False)

# Initialize database
init_database()

# Custom CSS for the blue theme
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

.main-header {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    color: white;
    text-align: center;
}

.auth-container {
    background: rgba(255,255,255,0.95);
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    max-width: 800px;
    margin: 0 auto;
}

.sidebar-style {
    background: linear-gradient(180deg, #4a5fc7 0%, #5b6bd6 100%);
    border-radius: 15px;
    padding: 20px;
}
"""

# Create Gradio Interface
with gr.Blocks(title="SpendLens-AI", theme=gr.themes.Soft(), css=custom_css) as app:
    # Session state
    user_session = gr.State(None)

    # Header
    with gr.Row():
        gr.HTML("""
            <div class="main-header">
                <h1>💰 SpendLens-AI</h1>
                <p>Smart Receipt Analysis & Expense Tracking</p>
            </div>
        """)

    # Authentication Section (Visible by default)
    with gr.Group(visible=True) as auth_section:
        with gr.Row():
            gr.HTML('<div class="auth-container">')
            with gr.Column():
                gr.Markdown("# 🔐 Welcome to SpendLens-AI")
                gr.Markdown("Please login or create an account to continue")

                with gr.Tab("Login"):
                    login_username = gr.Textbox(label="👤 Username", placeholder="Enter username")
                    login_password = gr.Textbox(label="🔒 Password", type="password", placeholder="Enter password")
                    login_btn = gr.Button("🚀 Login", variant="primary", size="lg")
                    login_output = gr.Textbox(label="Status", interactive=False)

                with gr.Tab("Sign Up"):
                    signup_username = gr.Textbox(label="👤 Username", placeholder="Choose username")
                    signup_password = gr.Textbox(label="🔒 Password", type="password", placeholder="Choose password")
                    signup_confirm = gr.Textbox(label="🔒 Confirm Password", type="password", placeholder="Confirm password")
                    signup_btn = gr.Button("📝 Create Account", variant="secondary", size="lg")
                    signup_output = gr.Textbox(label="Status", interactive=False)
            gr.HTML('</div>')

    # Main Application (Hidden by default)
    with gr.Group(visible=False) as main_section:
        with gr.Row():
            # Sidebar
            with gr.Column(scale=1):
                gr.HTML("""
                    <div style="background: linear-gradient(180deg, #4a5fc7 0%, #5b6bd6 100%);
                                padding: 20px; border-radius: 15px; color: white; text-align: center;">
                        <h2>💰 SpendLens-AI</h2>
                        <div style="margin: 20px 0;">
                            <div id="user-avatar" style="width: 60px; height: 60px; background: rgba(255,255,255,0.2);
                                 border-radius: 50%; margin: 0 auto 10px; display: flex; align-items: center;
                                 justify-content: center; font-size: 24px;">👤</div>
                            <p style="margin: 0; opacity: 0.9;">Welcome Back!</p>
                        </div>
                    </div>
                """)
                logout_btn = gr.Button("🚪 Logout", variant="secondary")

            # Main Content
            with gr.Column(scale=3):
                with gr.Tabs():
                    # Dashboard Tab
                    with gr.Tab("🏠 Dashboard"):
                        gr.Markdown("## Dashboard")

                        with gr.Row():
                            with gr.Column():
                                dashboard_stats = gr.HTML()
                                refresh_dashboard = gr.Button("🔄 Refresh Dashboard", variant="secondary")

                        gr.Markdown("## 📋 Recent Receipts")
                        recent_receipts_table = gr.Dataframe(
                            headers=["Store", "Date", "Amount"],
                            datatype=["str", "str", "str"],
                            interactive=False
                        )

                    # Upload Receipt Tab
                    with gr.Tab("📷 Upload Receipt"):
                        gr.Markdown("## Upload New Receipt")

                        with gr.Row():
                            with gr.Column():
                                receipt_image = gr.Image(
                                    label="📸 Drop your receipt here or click to browse",
                                    type="pil",
                                    height=300
                                )
                                process_btn = gr.Button("🔍 Process Receipt", variant="primary", size="lg")

                            with gr.Column():
                                process_status = gr.Textbox(label="Status", interactive=False)
                                receipt_summary = gr.Markdown()

                        gr.Markdown("## 📊 Receipt Items")
                        receipt_items_table = gr.Dataframe(
                            headers=["Item Name", "Category", "Quantity", "Unit Price", "Total Price"],
                            datatype=["str", "str", "number", "str", "str"],
                            interactive=False
                        )

                    # Analytics Tab
                    with gr.Tab("📊 Analytics"):
                        gr.Markdown("## Monthly Spending Analysis")

                        with gr.Row():
                            year_input = gr.Number(label="Year", value=datetime.datetime.now().year, precision=0)
                            month_input = gr.Dropdown(
                                label="Month",
                                choices=[(calendar.month_name[i], i) for i in range(1, 13)],
                                value=datetime.datetime.now().month
                            )
                            show_calendar_btn = gr.Button("📈 Generate Report", variant="primary")

                        monthly_output = gr.Markdown()

    # Event handlers
    login_btn.click(
        handle_login,
        inputs=[login_username, login_password],
        outputs=[login_output, user_session, auth_section, main_section]
    )

    signup_btn.click(
        handle_signup,
        inputs=[signup_username, signup_password, signup_confirm],
        outputs=[signup_output, user_session, auth_section, main_section]
    )

    logout_btn.click(
        logout_user,
        outputs=[user_session, auth_section, main_section]
    )

    refresh_dashboard.click(
        show_dashboard,
        inputs=[user_session],
        outputs=[dashboard_stats, recent_receipts_table]
    )

    process_btn.click(
        process_receipt,
        inputs=[receipt_image, user_session],
        outputs=[process_status, receipt_items_table, receipt_summary]
    )

    show_calendar_btn.click(
        show_monthly_calendar,
        inputs=[year_input, month_input, user_session],
        outputs=[monthly_output]
    )

    # Load dashboard on app start for logged-in users
    app.load(
        show_dashboard,
        inputs=[user_session],
        outputs=[dashboard_stats, recent_receipts_table]
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=True, debug=True)
