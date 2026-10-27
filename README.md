# Automatic Text Summarization Using NLP and TextRank

## Project overview
This project is a local Streamlit web application for **extractive text summarization**. It uses TF-IDF sentence vectors, cosine similarity, and a PageRank-style TextRank scoring method to select key sentences.

## Features
- Paste text or upload a `.txt` article.
- Choose Short, Medium, or Long summary.
- View original and summary word counts, compression percentage, and number of selected sentences.
- Inspect sentence ranking scores.
- Download the summary as a text file.
- Evaluate a generated summary against a human-written reference using ROUGE-1, ROUGE-2, and ROUGE-L F1.
- No API key, paid service, or training dataset is required for the core algorithm.

## Requirements
Python 3.10–3.12 recommended. An internet connection is needed to install libraries.

## Windows 11 setup (VS Code terminal)
1. Extract the ZIP.
2. Open the extracted `automatic_text_summarization_upgraded` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Confirm files are present with `dir`. You should see `app.py`, `summarizer.py`, and `requirements.txt`.
5. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
6. Activate it in PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, choose Command Prompt in the terminal dropdown and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
7. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
8. Launch:
   ```powershell
   python -m streamlit run app.py
   ```
9. Open the local address printed in the terminal, usually `http://localhost:8501`.

## Automated tests
With the virtual environment active:
```powershell
python -m unittest discover -s tests -v
```

## Algorithm
1. Split the input into sentences.
2. Represent each sentence with TF-IDF.
3. Calculate cosine similarity between sentence vectors.
4. Treat each sentence as a graph node and similarities as weighted edges.
5. Calculate PageRank-style TextRank scores.
6. Select top-ranked sentences and restore their original order.

## ROUGE evaluation
The evaluation tab calculates single-reference ROUGE-1, ROUGE-2, and ROUGE-L F1 scores. ROUGE is a lexical overlap measure. Scores depend on the chosen reference summary and should not be interpreted as a complete measure of factual correctness, readability, or semantic equivalence.

For a basic faculty submission, test several different articles and report the input length, output length, compression ratio, ROUGE scores where reference summaries are available, and a brief qualitative observation. Do not claim results that you have not measured.

## Suggested 3–4 day plan
- Day 1: Run the application, test sample texts, understand the algorithm.
- Day 2: Test 5–10 articles and collect results; use reference summaries for ROUGE.
- Day 3: Prepare flowchart, screenshots, methodology, results, limitations and conclusion.
- Day 4: Re-test, proofread the report and practise viva answers.

## Limitations
- Extractive, not abstractive: output sentences come from the source.
- The lightweight sentence splitter may fail on abbreviations or unusual punctuation.
- ROUGE overlap scores are not a complete quality measure.
- Performance should be evaluated on multiple examples.
