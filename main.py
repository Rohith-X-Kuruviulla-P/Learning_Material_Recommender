import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import docx
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import io

app = FastAPI(
    title="BAWE Essay Recommender",
    description="API to find similar essays from the BAWE corpus.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_data: Dict[str, Any] = {}

BAWE_ROOT_PATH = r'./BAWE_dataset'

class TextInput(BaseModel):
    text: str

class Recommendation(BaseModel):
    id: str
    discipline: str
    level: str
    similarity_score: float

def extract_text_from_docx(file_like_object):
    try:
        doc = docx.Document(io.BytesIO(file_like_object.read()))
        full_text = [p.text for p in doc.paragraphs]
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error reading the document: {e}"

def load_bawe_corpus(root_dir):
    corpus_docs = []
    print(f"Loading files and using folder names for metadata from: {root_dir}")
    
    if not os.path.exists(root_dir):
        print(f"FATAL ERROR: The path '{root_dir}' does not exist.")
        return pd.DataFrame()

    for dirpath, _, filenames in tqdm(os.walk(root_dir), desc="Processing Disciplines"):
        for filename in filenames:
            if filename.endswith(".csv"):
                try:
                    file_path = os.path.join(dirpath, filename)
                    
                    parts = file_path.split(os.sep)
                    discipline = parts[-3]
                    level = parts[-2]
                    file_id = filename.replace('.csv', '')
                    
                    word_df = pd.read_csv(file_path)
                    lemmas_string = " ".join(word_df['lemma'].astype(str).dropna())
                    
                    corpus_docs.append({
                        'id': file_id,
                        'discipline': discipline,
                        'level': level,
                        'lemmas': lemmas_string
                    })
                except Exception as e:
                    print(f"Could not process file {filename}: {e}")
                    
    print("Corpus loading complete.")
    return pd.DataFrame(corpus_docs)

def process_new_text(text: str, nlp_model):
    doc = nlp_model(text)
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return " ".join(lemmas)

def get_recommendations(lemmas: str) -> List[Dict]:
    new_tfidf = model_data["vectorizer"].transform([lemmas])
    
    cosine_similarities = cosine_similarity(new_tfidf, model_data["corpus_tfidf_matrix"])
    similarity_scores = cosine_similarities[0]
    
    top_5_indices = similarity_scores.argsort()[-5:][::-1]

    recommendations_df = model_data["corpus_df"].iloc[top_5_indices].copy()
    
    recommendations_df['similarity_score'] = similarity_scores[top_5_indices]

    return recommendations_df.to_dict(orient="records")

@app.on_event("startup")
async def load_models():
    print("--- Server is starting up... ---")
    
    print("Loading spaCy model (en_core_web_sm)...")
    model_data["nlp"] = spacy.load("en_core_web_sm")
    
    print(f"Loading BAWE corpus from {BAWE_ROOT_PATH}...")
    corpus_df = load_bawe_corpus(BAWE_ROOT_PATH)
    if corpus_df.empty:
        print("FATAL: Corpus is empty. Server may not function correctly.")
    model_data["corpus_df"] = corpus_df
    
    print("Fitting TF-IDF Vectorizer on corpus...")
    model_data["vectorizer"] = TfidfVectorizer()
    model_data["corpus_tfidf_matrix"] = model_data["vectorizer"].fit_transform(corpus_df['lemmas'])
    
    print("--- Startup complete. Server is ready. ---")

@app.get("/")
async def root():
    return {"message": "BAWE Essay Recommender API. Use /docs for documentation."}

@app.post("/analyze-text/", response_model=List[Recommendation])
async def analyze_text(item: TextInput):
    your_lemmas = process_new_text(item.text, model_data["nlp"])
    
    recommendations = get_recommendations(your_lemmas)
    
    return recommendations

@app.post("/analyze-document/", response_model=List[Recommendation])
async def analyze_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .docx file.")
    
    document_content = extract_text_from_docx(file.file)
    
    if "Error" in document_content:
        raise HTTPException(status_code=500, detail=f"Error reading document: {document_content}")
        
    your_lemmas = process_new_text(document_content, model_data["nlp"])
    
    recommendations = get_recommendations(your_lemmas)
    
    return recommendations

if __name__ == "__main__":
    print("Starting Uvicorn server at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)