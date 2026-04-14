# 📄 AI-Powered Resume Analyzer & Multi-Source Job Matcher

An AI-powered web app that:

- Analyzes your uploaded resume (PDF)
- Fetches live jobs from multiple sources
- Matches the best job opportunities based on your skills and experience

Built using **Streamlit**, **Python**, **Pandas**, **Sentence-Transformers**, and **Machine Learning** for smart, semantic matching. [file:1][file:4]

---

## 🔥 Features

- 📄 Upload your Resume (PDF)
- 🧠 Automatic Resume Text Extraction (PDFMiner)
- 🌎 Fetch Jobs from Multiple Sources
  - RemoteOK (API)
  - Microsoft Careers
  - AngelList / Wellfound [file:4]
- 🤖 Smart Resume-to-Job Matching
  - Text preprocessing (HTML stripping, stopword removal, stemming)
  - Sentence-transformer embeddings (`all-MiniLM-L6-v2`)
  - Cosine similarity to rank best matches [file:4]
- 🎯 Job Filters (in sidebar)
  - Only Remote Jobs
  - Only Microsoft Jobs
  - Only Startups (AngelList)
  - Filter by Tech Stack (e.g., Python, React) [file:2]
- 🌗 Dark Mode UI with modern job cards (title, company, similarity score, Apply button) [file:2]

---

## 📥 Installation

1. **Clone the repository**

```
git clone https://github.com/umarkabir2007-sys/AI-Powered-Resume-Analyzer-Job-Matcher.git
```

2. **Create & activate virtual environment (recommended)**

```
python -m venv venv
# Windows (PowerShell)
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate
```

3. **Install dependencies**

```
pip install -r requirements.txt
```

This installs Streamlit, Pandas, scikit-learn, pdfminer.six, BeautifulSoup, sentence-transformers, torch, and NLTK. [file:3][file:4]

4. **Download NLTK data (once)**

```
python
>>> import nltk
>>> nltk.download('stopwords')
>>> exit()
```

---

## 🚀 Running the App

From the project folder (with venv active):

```
streamlit run app.py
```

- This starts a local server (usually `http://localhost:8501`). [file:2]
- In the browser:
  - Upload your resume (PDF).
  - The app extracts text, fetches jobs, matches them, and shows top job cards with similarity scores and “Apply Now” links. [file:2][file:4]

---

## 🛠 Project Structure

```
.
├── app.py          # Main Streamlit app (UI, filters, rendering)
├── utils.py        # Resume extraction, job fetching, matching logic
├── job_scraper.py  # Optional: scrape Indeed + insert into SQLite
├── database.py     # Optional: SQLAlchemy engine/table for jobs.db
├── init_db.py      # Optional: initialize jobs.db schema
├── jobs.csv        # Optional: sample/static jobs data
├── requirements.txt
└── README.md
```

[files:2][file:4][file:5][file:6][file:7]

---

## ✨ Future Improvements

- Better salary estimation
- Apply to jobs with a single click
- AI-based resume improvement suggestions
- Richer scraping of full job descriptions from Microsoft/AngelList pages

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to open an Issue or Pull Request.

---

## 📄 License

This project is licensed under the **MIT License**.
```

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/133318752/55bd2f7a-866d-453d-a145-f804750372e4/README.md)
