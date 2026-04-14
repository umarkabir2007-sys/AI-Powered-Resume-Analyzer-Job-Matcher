import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download NLTK stopwords automatically (important for cloud deployment)
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))

stemmer = SnowballStemmer("english")
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_resume_text(pdf_file):
    """Extract raw text from an uploaded PDF file."""
    return extract_text(pdf_file)

def preprocess(text):
    """Clean and normalize text: strip HTML, lowercase, remove stopwords, stem."""
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def fetch_remoteok_jobs():
    """Fetch jobs from RemoteOK API."""
    url = "https://remoteok.com/api"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        jobs = []
        for item in data:
            if isinstance(item, dict) and item.get("position") and item.get("description") and item.get("url"):
                job_url = item.get("url", "")
                if not job_url.startswith("https://"):
                    job_url = "https://remoteok.com" + job_url
                jobs.append({
                    "title": item.get("position", ""),
                    "company": item.get("company", ""),
                    "description": item.get("description", ""),
                    "url": job_url,
                    "remote": True,
                    "source": "RemoteOK"
                })
        return pd.DataFrame(jobs)
    except Exception as e:
        print(f"RemoteOK error: {e}")
        return pd.DataFrame()

def fetch_microsoft_jobs():
    """Fetch jobs from Microsoft Careers (simple scraper)."""
    url = "https://careers.microsoft.com/us/en/search-results"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        jobs = []
        for job_card in soup.find_all("section", class_="jobs-list-container"):
            title_tag = job_card.find("h3")
            if title_tag:
                title = title_tag.text.strip()
                link_tag = job_card.find("a")
                if link_tag:
                    link = link_tag["href"]
                    full_link = "https://careers.microsoft.com" + link if link.startswith("/") else link
                    jobs.append({
                        "title": title,
                        "company": "Microsoft",
                        "description": "Details on Microsoft Career Page",
                        "url": full_link,
                        "remote": False,
                        "source": "Microsoft"
                    })
        return pd.DataFrame(jobs)
    except Exception as e:
        print(f"Microsoft error: {e}")
        return pd.DataFrame()

def fetch_angellist_jobs():
    """Fetch jobs from AngelList/Wellfound."""
    url = "https://wellfound.com/jobs"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        jobs = []
        job_cards = soup.find_all("div", class_="styles_component__P6AY4")
        for card in job_cards:
            title_tag = card.find("h2")
            company_tag = card.find("h3")
            link_tag = card.find("a", href=True)
            if title_tag and link_tag:
                title = title_tag.text.strip()
                company = company_tag.text.strip() if company_tag else "Startup"
                link = link_tag["href"]
                if not link.startswith("https://"):
                    link = "https://wellfound.com" + link
                jobs.append({
                    "title": title,
                    "company": company,
                    "description": "Details on AngelList Job Page",
                    "url": link,
                    "remote": False,
                    "source": "AngelList"
                })
        return pd.DataFrame(jobs)
    except Exception as e:
        print(f"AngelList error: {e}")
        return pd.DataFrame()

def fetch_all_jobs():
    """Fetch jobs from all sources and combine into one DataFrame."""
    remoteok = fetch_remoteok_jobs()
    microsoft = fetch_microsoft_jobs()
    angellist = fetch_angellist_jobs()
    all_jobs = pd.concat([remoteok, microsoft, angellist], ignore_index=True)
    if all_jobs.empty:
        return all_jobs
    all_jobs['description'] = all_jobs['description'].fillna("")
    all_jobs = all_jobs[all_jobs['description'].str.len() > 20].reset_index(drop=True)
    return all_jobs

def match_resume_to_jobs(resume_text, jobs_df):
    """Match resume to jobs using sentence embeddings and cosine similarity."""
    if jobs_df.empty:
        return pd.DataFrame(columns=["title", "company", "similarity", "url"])
    resume_clean = preprocess(resume_text)
    jobs_df = jobs_df.copy()
    jobs_df["processed"] = jobs_df["description"].apply(preprocess)
    if not resume_clean.strip():
        return pd.DataFrame(columns=["title", "company", "similarity", "url"])
    resume_emb = model.encode([resume_clean])
    job_embs = model.encode(jobs_df["processed"].tolist())
    sims = cosine_similarity(resume_emb, job_embs).flatten()
    jobs_df["similarity"] = sims
    top_matches = jobs_df.sort_values(by="similarity", ascending=False).head(10)
    return top_matches[["title", "company", "similarity", "url"]]
