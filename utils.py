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


# -----------------------------
# Global NLP / embedding setup
# -----------------------------

# Load stopwords & stemmer
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

# Load a lightweight sentence transformer model
# This will be downloaded the first time and then cached.
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Basic utilities
# -----------------------------

def extract_resume_text(pdf_file):
    """
    Extract raw text from an uploaded PDF file.
    """
    return extract_text(pdf_file)


def preprocess(text):
    """
    Clean and normalize text:
    - Handle non-string input.
    - Strip HTML.
    - Lowercase.
    - Remove non-letter characters.
    - Remove stopwords and short tokens.
    - Apply stemming.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    # Lowercase
    text = text.lower()

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords & very short tokens, apply stemming
    tokens = [
        stemmer.stem(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]

    return " ".join(tokens)


# -----------------------------
# Job fetching functions
# -----------------------------

def fetch_remoteok_jobs():
    """
    Fetch jobs from RemoteOK API.
    Returns a DataFrame with columns: title, company, description, url.
    """
    url = "https://remoteok.com/api"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    data = response.json()

    jobs = []
    for item in data:
        if (
            isinstance(item, dict)
            and item.get("position")
            and item.get("description")
            and item.get("url")
        ):
            job_url = item.get("url", "")
            if not job_url.startswith("https://"):
                job_url = "https://remoteok.com" + job_url

            jobs.append(
                {
                    "title": item.get("position", ""),
                    "company": item.get("company", ""),
                    "description": item.get("description", ""),
                    "url": job_url,
                }
            )

    return pd.DataFrame(jobs)


def fetch_microsoft_jobs():
    """
    Fetch jobs from Microsoft Careers (basic HTML scraping).
    WARNING: Descriptions are placeholder and not very informative.
    """
    url = "https://careers.microsoft.com/us/en/search-results"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, "html.parser")
    jobs = []

    for job_card in soup.find_all("section", class_="jobs-list-container"):
        title_tag = job_card.find("h3")
        if title_tag:
            title = title_tag.text.strip()
            link_tag = job_card.find("a")
            if link_tag:
                link = link_tag["href"]
                full_link = "https://careers.microsoft.com" + link
                jobs.append(
                    {
                        "title": title,
                        "company": "Microsoft",
                        "description": "Details on Microsoft Career Page",
                        "url": full_link,
                    }
                )

    return pd.DataFrame(jobs)


def fetch_angellist_jobs():
    """
    Fetch jobs from AngelList / Wellfound.
    WARNING: Descriptions are placeholder here too.
    """
    url = "https://wellfound.com/jobs"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    jobs = []
    job_cards = soup.find_all("div", class_="styles_component__P6AY4")  # CSS class used in your original code

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

            jobs.append(
                {
                    "title": title,
                    "company": company,
                    "description": "Details on AngelList Job Page",
                    "url": link,
                }
            )

    return pd.DataFrame(jobs)


def fetch_all_jobs():
    """
    Fetch jobs from all sources and concatenate into a single DataFrame.
    Also:
    - Drops rows with very short descriptions (<= 50 chars) to improve matching quality.
    """
    # RemoteOK is primary source with richer descriptions
    remoteok = fetch_remoteok_jobs()

    # Try Microsoft
    try:
        microsoft = fetch_microsoft_jobs()
    except Exception as e:
        print(f"Error fetching Microsoft jobs: {e}")
        microsoft = pd.DataFrame(columns=["title", "company", "description", "url"])

    # Try AngelList
    try:
        angellist = fetch_angellist_jobs()
    except Exception as e:
        print(f"Error fetching AngelList jobs: {e}")
        angellist = pd.DataFrame(columns=["title", "company", "description", "url"])

    all_jobs = pd.concat([remoteok, microsoft, angellist], ignore_index=True)

    # Filter out jobs with too short / generic descriptions
    all_jobs["description"] = all_jobs["description"].fillna("")
    all_jobs = all_jobs[all_jobs["description"].str.len() > 50].reset_index(drop=True)

    return all_jobs


# -----------------------------
# Matching logic
# -----------------------------

def match_resume_to_jobs(resume_text, jobs_df):
    """
    Match resume text to jobs using:
    - Preprocessing for both resume and job descriptions.
    - Sentence-transformer embeddings.
    - Cosine similarity.
    Returns top 10 matches with title, company, similarity, and url.
    """
    if jobs_df.empty:
        return pd.DataFrame(columns=["title", "company", "similarity", "url"])

    # Clean resume and job descriptions
    resume_clean = preprocess(resume_text)
    jobs_df = jobs_df.copy()
    jobs_df["processed"] = jobs_df["description"].apply(preprocess)

    # If everything becomes empty after preprocessing, avoid crashing
    if not resume_clean.strip():
        return pd.DataFrame(columns=["title", "company", "similarity", "url"])

    # Encode resume and jobs using sentence-transformer model
    resume_emb = model.encode([resume_clean])
    job_embs = model.encode(jobs_df["processed"].tolist())

    sims = cosine_similarity(resume_emb, job_embs).flatten()
    jobs_df["similarity"] = sims

    # Sort and pick top 10
    top_matches = jobs_df.sort_values(by="similarity", ascending=False).head(10)

    return top_matches[["title", "company", "similarity", "url"]]
