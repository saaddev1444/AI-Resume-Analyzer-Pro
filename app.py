import streamlit as st
import PyPDF2
import re
import io
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF

# Page Configuration
st.set_page_config(
    page_title="AI Resume Analyzer Pro",
    page_icon="📄",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title { text-align: center; color: #1E88E5; font-weight: bold; }
    .sub-title { text-align: center; color: #555; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>📄 AI Resume Analyzer & ATS Matcher Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Advanced NLP Multi-Category Skill Analysis & Actionable ATS Insights</p>", unsafe_allow_html=True)

# Helper Function: Extract Text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text

# Helper Function: Text Cleaning
def clean_text(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# Helper Function to sanitize text for FPDF Standard Latin-1
def encode_for_pdf(text):
    return str(text).encode('latin-1', 'replace').decode('latin-1')

# Keyword Dictionaries for Hard vs Soft Skills & Categories
SKILL_CATEGORIES = {
    "Programming Languages": ["python", "java", "cpp", "c", "javascript", "typescript", "html", "css", "sql", "r"],
    "Data Science & ML": ["pandas", "numpy", "scikit", "learn", "matplotlib", "seaborn", "tensorflow", "pytorch", "keras", "opencv", "eda", "classification", "regression"],
    "Tools & Platforms": ["git", "github", "docker", "vscode", "jupyter", "streamlit", "netlify", "aws", "azure", "linux"],
    "Soft Skills": ["communication", "leadership", "teamwork", "problem", "solving", "management", "analytical", "creativity"]
}

SOFT_SKILLS_SET = set(SKILL_CATEGORIES["Soft Skills"])

# Helper Function: Native FPDF Chart Generator (Guarantees PDF Visuals without Kaleido crashes)
def draw_native_pdf_charts(pdf, score, category_scores):
    # Section Title
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 8, txt=encode_for_pdf("Visual ATS Match Analytics:"), ln=True)
    pdf.ln(2)

    # 1. Visual Progress Bar for Overall Score
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(190, 5, txt=encode_for_pdf(f"Overall ATS Score Gauge: {score}%"), ln=True)
    
    # Draw Background Bar
    pdf.set_fill_color(230, 230, 230)
    pdf.rect(10, pdf.get_y(), 180, 8, 'F')
    
    # Select Bar Color based on Score
    if score >= 70:
        pdf.set_fill_color(76, 175, 80) # Green
    elif score >= 40:
        pdf.set_fill_color(255, 193, 7) # Yellow/Orange
    else:
        pdf.set_fill_color(244, 67, 54) # Red
        
    fill_width = (score / 100.0) * 180
    pdf.rect(10, pdf.get_y(), fill_width, 8, 'F')
    pdf.ln(12)

    # 2. Category Match Breakdown Horizontal Bar Chart
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(190, 5, txt=encode_for_pdf("Category Wise Breakdown:"), ln=True)
    pdf.ln(2)

    for cat, cat_score in category_scores.items():
        pdf.set_font("Arial", size=9)
        pdf.cell(60, 6, txt=encode_for_pdf(f"{cat} ({cat_score}%):"), ln=False)
        
        current_y = pdf.get_y()
        # Draw background bar
        pdf.set_fill_color(240, 240, 240)
        pdf.rect(70, current_y + 1, 110, 4, 'F')
        
        # Draw value bar
        pdf.set_fill_color(30, 136, 229) # Blue
        cat_fill_width = (cat_score / 100.0) * 110
        if cat_fill_width > 0:
            pdf.rect(70, current_y + 1, cat_fill_width, 4, 'F')
        pdf.ln(6)
        
    pdf.ln(4)

# Helper Function: PDF Report Generator
def generate_pdf_report(score, matched, missing, word_count, category_scores, fig_gauge, fig_bar):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(190, 10, txt=encode_for_pdf("ATS Resume Matcher Analysis Report"), ln=True, align='C')
    pdf.ln(5)
    
    # Overview
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 8, txt=encode_for_pdf("Summary Overview:"), ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(190, 6, txt=encode_for_pdf(f"- Overall ATS Match Score: {score}%"), ln=True)
    pdf.cell(190, 6, txt=encode_for_pdf(f"- Total Resume Word Count: {word_count} words"), ln=True)
    pdf.ln(4)
    
    # Draw PDF Native Visual Charts
    draw_native_pdf_charts(pdf, score, category_scores)

    # Matched Keywords
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 8, txt=encode_for_pdf("Top Matched Keywords:"), ln=True)
    pdf.set_font("Arial", size=10)
    matched_text = ", ".join(matched[:25]) if matched else "None"
    pdf.multi_cell(190, 6, txt=encode_for_pdf(matched_text))
    pdf.ln(4)
    
    # Missing Keywords
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 8, txt=encode_for_pdf("Critical Missing Keywords to Add:"), ln=True)
    pdf.set_font("Arial", size=10)
    missing_text = ", ".join(missing[:25]) if missing else "None"
    pdf.multi_cell(190, 6, txt=encode_for_pdf(missing_text))
    
    return bytes(pdf.output())

# Extended Stopwords Filter List
CUSTOM_STOP_WORDS = list(TfidfVectorizer(stop_words='english').get_stop_words()) + [
    'should', 'tasks', 'including', 'candidates', 'candidate', 'knowledge', 
    'like', 'pursuing', 'have', 'must', 'join', 'team', 'preferred', 'required',
    'will', 'our', 'are', 'the', 'technologies', 'seeking', 'working',
    'ability', 'strong', 'good', 'work', 'years', 'job', 'description', 'role'
]

# Main Layout Input
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📌 1. Upload Resume")
    uploaded_resume = st.file_uploader("Upload PDF Resume", type=["pdf"], help="Upload single or multi-page PDF resume")
    
    # Upload Alert Notification
    if uploaded_resume is not None:
        st.success(f"✅ CV Uploaded Successfully! ({uploaded_resume.name})")

with col2:
    st.subheader("📌 2. Job Description")
    job_description = st.text_area("Paste Job Description (JD)", height=180, placeholder="Paste requirements, responsibilities, and key skills...")

st.markdown("---")

if st.button("🚀 Analyze & Match Resume Pro", use_container_width=True):
    if uploaded_resume is not None and job_description.strip() != "":
        with st.spinner("Executing Deep NLP Vectorization & Multi-Layer Skill Classification..."):
            
            # Extract & Clean Text
            raw_resume_text = extract_text_from_pdf(uploaded_resume)
            clean_resume = clean_text(raw_resume_text)
            clean_jd = clean_text(job_description)
            word_count = len(clean_resume.split())
            
            # TF-IDF & Cosine Similarity Score
            vectorizer = TfidfVectorizer(stop_words=CUSTOM_STOP_WORDS)
            tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            match_percentage = round(similarity[0][0] * 100, 2)
            
            # Token Extraction
            feature_names = vectorizer.get_feature_names_out()
            jd_tokens = set([w for w in clean_jd.split() if w not in CUSTOM_STOP_WORDS])
            resume_tokens = set([w for w in clean_resume.split() if w not in CUSTOM_STOP_WORDS])
            
            matched_keywords = [word for word in feature_names if word in jd_tokens and word in resume_tokens and len(word) > 2]
            missing_keywords = [word for word in jd_tokens if word not in resume_tokens and len(word) > 2 and not word.isdigit()]
            
            # Separate Hard vs Soft Skills
            matched_hard = [w for w in matched_keywords if w not in SOFT_SKILLS_SET]
            matched_soft = [w for w in matched_keywords if w in SOFT_SKILLS_SET]
            missing_hard = [w for w in missing_keywords if w not in SOFT_SKILLS_SET]
            missing_soft = [w for w in missing_keywords if w in SOFT_SKILLS_SET]

            # Category Wise Breakdown Analysis
            category_scores = {}
            for cat_name, cat_skills in SKILL_CATEGORIES.items():
                cat_jd_count = sum(1 for skill in cat_skills if re.search(r'\b' + re.escape(skill) + r'\b', clean_jd))
                cat_res_count = sum(1 for skill in cat_skills if re.search(r'\b' + re.escape(skill) + r'\b', clean_jd) and re.search(r'\b' + re.escape(skill) + r'\b', clean_resume))
                
                if cat_jd_count > 0:
                    category_scores[cat_name] = round((cat_res_count / cat_jd_count) * 100, 1)
                else:
                    category_scores[cat_name] = 100.0 if cat_res_count > 0 else 0.0

            # 1. Top Section: Gauge Score & Summary
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.subheader("📊 ATS Match Score")
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = match_percentage,
                    number = {'suffix': "%"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1E88E5"},
                        'steps': [
                            {'range': [0, 40], 'color': "#FFCDD2"},
                            {'range': [40, 70], 'color': "#FFE082"},
                            {'range': [70, 100], 'color': "#C8E6C9"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                st.subheader("📑 Match Summary")
                if match_percentage >= 70:
                    st.success("🎉 **Strong ATS Profile!** Your resume aligns exceptionally well with this job description.")
                elif match_percentage >= 40:
                    st.warning("⚠️ **Moderate Match!** Adding missing technical keywords will significantly improve your match score.")
                else:
                    st.error("❌ **Low Match!** Consider overhauling your resume to align closely with the targeted Job Description requirements.")
                
                st.write(f"📏 **Resume Word Count:** {word_count} words")
                st.write(f"⚙️ **Hard Skills Matched:** {len(matched_hard)} terms")
                st.write(f"🤝 **Soft Skills Matched:** {len(matched_soft)} terms")

            st.markdown("---")

            # 2. Skill Category Breakdown Chart
            st.subheader("🎯 Skill Category Breakdown")
            fig_bar = px.bar(
                x=list(category_scores.keys()),
                y=list(category_scores.values()),
                labels={'x': 'Skill Category', 'y': 'Match Percentage (%)'},
                color=list(category_scores.values()),
                color_continuous_scale="Blues",
                text=[f"{v}%" for v in category_scores.values()]
            )
            fig_bar.update_layout(height=300, yaxis_range=[0, 100])
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")

            # 3. Hard vs Soft Skills Filter Tabs
            st.subheader("🔍 Deep Keyword Analysis (Hard vs Soft Skills)")
            tab1, tab2, tab3 = st.tabs(["💻 Matched Hard Skills", "⚠️ Missing Hard Skills", "🤝 Soft Skills Analysis"])
            
            with tab1:
                st.success(f"Total **{len(matched_hard)}** Technical Terms Matched:")
                st.write(", ".join([f"`{kw}`" for kw in matched_hard[:30]]) if matched_hard else "No major technical skills matched.")

            with tab2:
                st.error(f"Total **{len(missing_hard)}** Critical Technical Terms Missing:")
                st.write(", ".join([f"`{kw}`" for kw in missing_hard[:30]]) if missing_hard else "All major technical terms are present.")

            with tab3:
                c1, c2 = st.columns(2)
                with c1:
                    st.info(f"**Matched Soft Skills ({len(matched_soft)}):**")
                    st.write(", ".join([f"`{kw}`" for kw in matched_soft]) if matched_soft else "None")
                with c2:
                    st.warning(f"**Missing Soft Skills ({len(missing_soft)}):**")
                    st.write(", ".join([f"`{kw}`" for kw in missing_soft]) if missing_soft else "None")

            st.markdown("---")

            # 4. Export Complete Analysis (PDF + Graphs)
            st.subheader("📥 Export Complete Analysis")
            pdf_bytes = generate_pdf_report(
                match_percentage, 
                matched_keywords, 
                missing_keywords, 
                word_count, 
                category_scores,
                fig_gauge,
                fig_bar
            )
            st.download_button(
                label="📄 Download Executive ATS Report (PDF with Charts)",
                data=pdf_bytes,
                file_name="Executive_ATS_Resume_Analysis.pdf",
                mime="application/pdf"
            )

    else:
        st.error("Please upload a PDF resume and paste the Job Description.")