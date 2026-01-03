# Hackathon Supervity 2026 🚀

## 📊 F1: Automated MD&A Draft from Financials (RAG + Summarization)

### 🎯 Problem Statement

**Objective:** Automatically generate first-draft MD&A (Management Discussion & Analysis) narratives from tabular financial statement extracts using AI-powered RAG (Retrieval Augmented Generation) and summarization techniques.

**What is MD&A?**  
Management Discussion & Analysis is a critical section in financial reports where executives explain:
- Financial performance trends
- Key business drivers and decisions
- Risks and uncertainties
- Future outlook

**Challenge:** Transform raw financial data into professional, coherent narratives that analysts and executives can review and refine.

---

### 📁 Dataset

**Source:** Financial Statement Extracts (SEC)  
**Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/securities-exchange-commission/financial-statement-extracts)

**Dataset includes:**
- Income statements, balance sheets, and cash flow statements
- Historical financial data from SEC filings
- Multiple companies and time periods

---

### 🎯 24-Hour Hackathon Deliverables

Build a complete solution (Jupyter Notebook + Python scripts) that performs the following:

#### 1️⃣ **Financial Data Processing & Analysis**
   - 📥 Load and parse financial statement extracts
   - 📈 Calculate Year-over-Year (YoY) growth/decline percentages
   - 📊 Compute Quarter-over-Quarter (QoQ) changes
   - 🔢 Generate key financial KPIs (margins, ratios, growth rates)

#### 2️⃣ **Document Processing & Vectorization**
   - 📄 Extract and chunk SEC filing documents
   - 🔍 Create embeddings for semantic search
   - 💾 Store in vector database for efficient retrieval

#### 3️⃣ **AI-Powered Narrative Generation**
   - 🤖 Generate structured MD&A sections using LLM:
     - **Executive Summary** - High-level overview
     - **Financial Performance** - Revenue, expenses, profitability trends
     - **Operational Highlights** - Key drivers and initiatives
     - **Risk Factors** - Challenges and uncertainties
     - **Forward Outlook** - Future expectations
   - 📎 Include citations linking back to source data/documents
   - ✅ Ensure factual accuracy with RAG approach

---

### 🛠️ Recommended Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.10+ | Core development |
| **Data Processing** | Pandas, NumPy | Financial data manipulation |
| **LLM Framework** | LangChain | RAG orchestration & prompting |
| **Embeddings** | OpenAI text-embedding-3-small | Document vectorization |
| **LLM Provider** | OpenAI GPT-4 / Gemini / Claude / Local LLMs | Text generation |
| **Vector Store** | ChromaDB or FAISS | Semantic search |
| **Schema/Validation** | Pydantic | Data validation & structuring |
| **Notebook** | Jupyter Lab | Interactive development |

---

### 📂 Project Structure

```
hackathon-supervity-2026/
│
├── 📄 README.md                    # Project documentation
├── 📋 GUIDELINES.md                # Hackathon guidelines
├── 📓 notebooks/                   
│   ├── 01_data_exploration.ipynb  # Initial data analysis
│   ├── 02_kpi_calculation.ipynb   # Financial metrics computation
│   └── 03_mda_generation.ipynb    # Full RAG pipeline
│
├── 🐍 scripts/                     
│   ├── data_processor.py          # Financial data processing
│   ├── vector_store.py            # Embedding & retrieval logic
│   └── mda_generator.py           # MD&A generation pipeline
│
├── 📊 data/                        
│   ├── raw/                       # Original SEC data
│   └── processed/                 # Cleaned & computed data
│
├── 📝 output/                      
│   └── generated_mdas/            # Generated MD&A drafts
│
├── 📦 requirements.txt             # Python dependencies
└── 🔧 .env                         # API keys (not in git)
```

---

### 🚀 Getting Started

#### **Prerequisites**
- Python 3.10 or higher
- Kaggle account (for dataset download)
- OpenAI API key (or alternative LLM access)

#### **Setup Instructions**

1. **Clone the repository**
   ```bash
   git clone https://github.com/PythonGuruGlobal/hackathon-supervity-2026.git
   cd hackathon-supervity-2026
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Go to [Kaggle Dataset](https://www.kaggle.com/datasets/securities-exchange-commission/financial-statement-extracts)
   - Download and extract to `data/raw/`

5. **Configure API keys**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

6. **Run the pipeline**
   ```bash
   # Option 1: Run Jupyter notebooks interactively
   jupyter lab notebooks/

   # Option 2: Run the complete script
   python scripts/mda_generator.py --input data/raw/financials.csv --output output/
   ```

---

### 💡 Key Features

✅ **Automated Financial Analysis** - Compute YoY, QoQ trends and KPIs  
✅ **RAG-Powered Generation** - Grounded narratives with source citations  
✅ **Multi-Section Output** - Structured MD&A with all key sections  
✅ **Factual Accuracy** - Retrieval ensures claims are backed by data  
✅ **Customizable Prompts** - Easily adapt narrative style and focus  

---

### 📈 Expected Output Example

**Input:** Financial statements for Company XYZ (Q3 2024)

**Generated MD&A Draft:**
```markdown
## Executive Summary
Company XYZ reported strong Q3 2024 results with revenue of $1.2B, 
representing 15% YoY growth driven by cloud services expansion...

## Financial Performance
- Revenue increased 15% YoY to $1.2B (Q3 2023: $1.04B) [Source: Income Statement, Q3 2024]
- Operating margin improved to 22% from 19% YoY [Source: Financial Ratios]
...
```

---

### 🏆 Success Criteria

- ✅ Successfully processes SEC financial data
- ✅ Calculates accurate financial metrics (YoY, QoQ, KPIs)
- ✅ Generates coherent, professional MD&A narratives
- ✅ Includes proper citations to source data
- ✅ Structured output (markdown with clear sections)
- ✅ Scalable to multiple companies/time periods

---

### 📚 Resources

- [SEC Financial Reporting Guide](https://www.sec.gov/reportspubs/investor-publications/investorpubsbegfinstmtguidehtm.html)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

---

### 📝 License

MIT License

---

### 👥 Contributing

This is a hackathon project. Feel free to fork and improve!

---

**⏱️ Hackathon Duration:** 24 hours  
**🎯 Goal:** Automate financial narrative generation with RAG-based AI summarization  
**🏅 Challenge:** Transform raw data into executive-ready insights
