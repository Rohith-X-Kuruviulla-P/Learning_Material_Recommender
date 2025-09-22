import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import docx
nlp = spacy.load("en_core_web_sm")



def extract_text_from_docx(file_path):
    """
    Extracts text from a .docx file and returns it as a single string.
    """
    try:
        doc = docx.Document(file_path)
        full_text = [p.text for p in doc.paragraphs]
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error reading the document: {e}"

#inserting input doc here
file_path = ''
document_content = extract_text_from_docx(file_path)

if "Error" not in document_content:
    print("--- Extracted Content ---")
    print(document_content)
else:
    print(document_content)

def load_bawe_corpus(root_dir):
    """
    Walks through the BAWE directory, using the folder names for metadata.
    """
    corpus_docs = []
    print(f"Loading files and using folder names for metadata from: {root_dir}")
    
    if not os.path.exists(root_dir):
        print(f"FATAL ERROR: The path '{root_dir}' does not exist. Please check the BAWE_ROOT_PATH variable.")
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
    #print(f"Total documents loaded: {corpus_docs}")
    return pd.DataFrame(corpus_docs)

def process_new_text(text):
    """
    Processes raw input text to extract lemmas.
    """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop])

#Load the BAWE Dataset
BAWE_ROOT_PATH = r'./BAWE_dataset'

corpus_df = load_bawe_corpus(BAWE_ROOT_PATH)


if not corpus_df.empty:
    your_input_text = document_content
    
    your_lemmas = process_new_text(your_input_text)
    
    # --- 3. COMBINE AND VECTORIZE ---
    all_docs_df = corpus_df.copy()
    input_doc = {"id": "your_input", "discipline": "N/A", "level": "N/A", "lemmas": your_lemmas}
    all_docs_df = pd.concat([all_docs_df, pd.DataFrame([input_doc])], ignore_index=True)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_docs_df['lemmas'])

    # --- 4. CALCULATE SIMILARITY ---
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix)
    similarity_scores = cosine_similarities[0][:-1]

    # --- 5. GENERATE RECOMMENDATIONS ---
    corpus_df['similarity_score'] = similarity_scores
    recommendations = corpus_df.sort_values(by='similarity_score', ascending=False)

    print("\n--- Your Input Text ---")
    print(your_input_text)
    print("\n--- Top 5 Recommended BAWE Essays ---")
    print(recommendations[['id', 'discipline', 'level', 'similarity_score']].head(5).to_string(index=False))
else:

    print("\nCould not generate recommendations because the corpus failed to load.")
