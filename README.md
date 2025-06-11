# SpendLens-AI: Smart Receipt Analysis & Expense Tracking

![SpendLens-AI Demo](demo.gif)

## 📌 Overview

SpendLens-AI is an intelligent receipt processing application that uses Optical Character Recognition (OCR) and Large Language Models (LLMs) to extract and analyze receipt data. The application helps users track their spending patterns, categorize purchases, and gain insights into their financial habits.

## Screenshots
![image alt](https://github.com/aniq63/SpendLens-AI/blob/48564b8f79931593faac639a6d605d648fd409ff/Capture.PNG)
![image alt](https://github.com/aniq63/SpendLens-AI/blob/8dc62e06885bec3d24ba52d3081c8c39ffba9e0a/screenshot1.PNG)
![image alt](https://github.com/aniq63/SpendLens-AI/blob/083e002042bfa0368ef362c77f96e08016546ffa/screenshot1.PNG)


## ✨ Key Features

- **📸 Receipt Processing**: Upload images of receipts for automatic data extraction
- **🔍 AI-Powered Analysis**: Uses OCR (Tesseract) and LLM (Groq/Llama 3) for accurate data extraction
- **📊 Expense Tracking**: Categorizes purchases and tracks spending over time
- **📈 Visual Analytics**: Provides spending insights with interactive dashboards
- **🔐 User Authentication**: Secure login system with password hashing
- **💾 Data Persistence**: SQLite database for storing all receipt data

## 🛠️ Technology Stack

- **Backend**: Python
- **LLM Integration**: Groq API with Llama 3 model
- **OCR**: Tesseract OCR
- **Database**: SQLite
- **Web Interface**: Gradio
- **Additional Libraries**: 
  - LangChain for LLM orchestration
  - Pydantic for data validation
  - Pillow for image processing

## 📋 Requirements

### System Requirements
- Python 3.8+
- Tesseract OCR installed on your system

### Python Packages
All required Python packages are listed in `requirements.txt`.

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aniq63/SpendLens-AI.git
   cd SpendLens-AI
   ```

2. **Install Tesseract OCR**:
   - On Ubuntu/Debian:
     ```bash
     sudo apt install tesseract-ocr
     ```
   - On macOS (using Homebrew):
     ```bash
     brew install tesseract
     ```
   - On Windows: Download installer from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

## � Usage

1. **Run the application**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   - The application will start a local server (usually at `http://127.0.0.1:7860`)
   - Open this URL in your web browser

3. **Using the application**:
   - Create an account or log in
   - Upload receipt images via the "Upload Receipt" tab
   - View your spending analytics in the "Dashboard" and "Analytics" tabs

## 📂 File Structure

```
SpendLens-AI/
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── packages.txt          # System packages (for reference)
├── README.md             # This file
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.
