# 📄 AI Resume Analyzer & ATS Matcher Pro

An advanced NLP-powered web application that analyzes PDF resumes against job descriptions, providing detailed ATS match scores, multi-category skill breakdowns, hard vs. soft skill classifications, and downloadable executive PDF reports with visual analytics.

---

## 🌟 Key Features

* **📄 PDF Text Extraction:** Parses text seamlessly from single or multi-page PDF resumes.
* **🤖 Deep NLP Vectorization:** Uses `TfidfVectorizer` and Cosine Similarity for precise mathematical textual match analysis.
* **📊 Visual ATS Match Gauge:** Displays an interactive gauge chart highlighting overall job alignment.
* **🎯 Multi-Category Skill Breakdown:** Evaluates match percentages across specific domains:
  * Programming Languages
  * Data Science & ML
  * Tools & Platforms
  * Soft Skills
* **🔍 Hard vs. Soft Skills Categorization:** Separates technical keywords from interpersonal capabilities for actionable resume tuning.
* **📄 Executive PDF Report Export:** Generates a clean PDF summary containing match breakdowns, key insights, and native visual progress indicators.

---

## 🛠️ Tech Stack & Libraries

* **Frontend & Web Framework:** Streamlit
* **PDF Parsing:** PyPDF2
* **NLP & Machine Learning:** Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`)
* **Data Visualizations:** Plotly Express & Plotly Graph Objects
* **PDF Report Generation:** FPDF
* **Pattern Matching:** Python Standard `re` (Regular Expressions)

---

## 🚀 Local Installation & Setup

Follow these steps to run the application locally on your machine:

1. **Clone the repository:**
   ```bash
 git clone https://github.com/saaddev1444/AI-Resume-Analyzer-Pro.git
cd AI-Resume-Analyzer-Pro