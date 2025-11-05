Project Description:

BAWE-based Essay Recommendation System

This project builds a content-based recommendation system for academic essays using the British Academic Written English (BAWE) corpus. The system compares a user-provided document (e.g., a .docx essay) or free-form text with essays in the BAWE dataset and recommends the most similar essays.

Dataset link: https://www.coventry.ac.uk/research/research-directories/current-projects/2015/british-academic-written-english-corpus-bawe/search-the-bawe-corpus/

Key Features

Document Input

Reads a .docx file (e.g., politics_and_life_science.docx) and extracts raw text.

Alternatively, allows the user to type in custom input text.

Text Processing

Uses spaCy (en_core_web_sm) to lemmatize words.

Removes punctuation and stop words.

Ensures consistent text representation for better similarity matching.

Corpus Loading

Loads BAWE essays stored as .csv files (grouped by discipline and academic level).

Extracts lemmas from each essay and organizes them with metadata (id, discipline, level).

TF-IDF Vectorization

Applies TF-IDF (scikit-learn) to convert all essays + the input text into numerical feature vectors.

Cosine Similarity Matching

Computes cosine similarity between the input text and all BAWE essays.

Scores each essay based on closeness of content.

Recommendations

Ranks BAWE essays in descending order of similarity.

Displays the Top 5 most relevant essays, along with their discipline, level, and similarity score.

🛠️ Tech Stack

Python

spaCy (NLP, lemmatization)

scikit-learn (TF-IDF, cosine similarity)

pandas (data handling)

tqdm (progress bar)

python-docx (Word document reading)

Workflow

Load and preprocess BAWE essays (.csv files).

Extract lemmas from user input (Word doc or typed text).

Combine BAWE corpus + user input for vectorization.

Compute similarity scores using TF-IDF + cosine similarity.

Output ranked recommendations.

Outcome:
A recommendation engine that helps students, researchers, or educators find similar academic essays in the BAWE corpus, given any new piece of writing.
