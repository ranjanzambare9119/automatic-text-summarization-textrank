import io
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from summarizer import summarize_text, count_words, evaluate_against_reference

st.set_page_config(page_title="Automatic Text Summarization", page_icon="📝", layout="wide")

st.title("Automatic Text Summarization Using NLP")
st.caption("Third-year engineering mini-project | Extractive summarization with TextRank")

with st.sidebar:
    st.header("Summarization settings")
    summary_length = st.select_slider(
        "Summary length",
        options=["Short", "Medium", "Long"],
        value="Medium",
    )
    st.markdown("---")
    st.markdown("**Core algorithm:** TextRank")
    st.markdown("**Pipeline:** Sentence splitting → TF-IDF → cosine similarity → graph ranking")
    st.markdown("---")
    st.caption("Runs locally. No API key is required.")

tab_summarize, tab_evaluate, tab_about = st.tabs(
    ["Summarize text", "Evaluate a summary", "How it works"]
)

with tab_summarize:
    st.subheader("Enter or upload an article")
    uploaded = st.file_uploader("Upload a plain-text article (.txt)", type=["txt"])
    default_text = """Machine learning is a branch of artificial intelligence that allows computer systems to identify patterns in data and make predictions. It is used in healthcare, finance, agriculture, education, and transportation. Before a model can learn, data scientists often clean the data, handle missing values, and select useful features.

Text summarization is an important application of natural language processing. It reduces a long document into a shorter version while attempting to preserve the main information. Extractive summarization selects sentences from the original document, while abstractive summarization generates new wording. Extractive methods are often easier to implement and explain.

TextRank is a graph-based ranking algorithm inspired by PageRank. In a TextRank summarizer, each sentence is represented as a node. Similarity between sentences determines the strength of the edges connecting them. Sentences with stronger relationships to other important sentences receive higher ranking scores.

This project converts sentences into TF-IDF vectors and calculates cosine similarity between each pair. It applies a PageRank-style calculation to the sentence similarity graph. The highest-ranked sentences are selected and then returned in their original order so the summary follows the source document.

The system provides different summary lengths and reports word counts and compression percentage. These measurements describe how much shorter the summary is, but they do not by themselves prove that the summary is accurate or coherent. Proper evaluation should also use reference summaries and human review."""
    if uploaded is not None:
        try:
            article = uploaded.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            article = uploaded.getvalue().decode("utf-8", errors="replace")
        st.text_area("Uploaded article preview", article, height=180, disabled=True)
    else:
        article = st.text_area("Paste text here", value=default_text, height=230)

    if st.button("Generate summary", type="primary"):
        if not article.strip():
            st.error("Enter or upload an article first.")
        else:
            summary, ranking = summarize_text(article, summary_length)
            st.session_state["article"] = article
            st.session_state["summary"] = summary
            st.session_state["ranking"] = ranking

    if "summary" in st.session_state and st.session_state.get("article") == article:
        summary = st.session_state["summary"]
        st.subheader("Generated summary")
        st.write(summary)
        original_words = count_words(article)
        summary_words = count_words(summary)
        compression = 0 if original_words == 0 else (1 - summary_words / original_words) * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original words", original_words)
        c2.metric("Summary words", summary_words)
        c3.metric("Compression", f"{compression:.1f}%")
        c4.metric("Sentences selected", len([x for x in re.split(r"(?<=[.!?])\s+", summary.strip()) if x]))
        st.download_button(
            "Download summary as .txt",
            data=summary,
            file_name="summary.txt",
            mime="text/plain",
        )
        with st.expander("Sentence ranking (TextRank scores)"):
            ranking_df = pd.DataFrame(st.session_state["ranking"])
            if not ranking_df.empty:
                ranking_df = ranking_df.rename(columns={"index": "Sentence number", "score": "TextRank score", "sentence": "Sentence"})
                st.dataframe(ranking_df, use_container_width=True, hide_index=True)
                st.caption("The summary uses the highest-ranked sentences, restored to their original document order.")

with tab_evaluate:
    st.subheader("Evaluate against a reference summary")
    st.write("Paste a human-written reference summary to calculate ROUGE-1, ROUGE-2, and ROUGE-L F1 scores. These scores measure text overlap; they do not fully measure factual correctness or readability.")
    reference = st.text_area("Reference summary", key="reference_summary", height=140)
    candidate = st.text_area("System-generated summary", key="candidate_summary", height=140,
                             value=st.session_state.get("summary", ""))
    if st.button("Calculate ROUGE scores"):
        if not reference.strip() or not candidate.strip():
            st.warning("Enter both a reference summary and a generated summary.")
        else:
            scores = evaluate_against_reference(candidate, reference)
            metrics = st.columns(3)
            metrics[0].metric("ROUGE-1 F1", f"{scores['rouge1_f1']:.3f}")
            metrics[1].metric("ROUGE-2 F1", f"{scores['rouge2_f1']:.3f}")
            metrics[2].metric("ROUGE-L F1", f"{scores['rougeL_f1']:.3f}")
            st.dataframe(pd.DataFrame([{"Metric": k, "F1 score": round(v, 4)} for k, v in scores.items()]),
                         use_container_width=True, hide_index=True)

with tab_about:
    st.subheader("System workflow")
    st.markdown("""
    1. **Sentence segmentation:** split the input document into sentences.
    2. **TF-IDF representation:** represent each sentence as a numerical vector.
    3. **Cosine similarity:** estimate similarity between every pair of sentences.
    4. **Sentence graph:** treat sentences as nodes and similarity values as edge weights.
    5. **TextRank:** apply PageRank-style iterative scoring to rank sentences.
    6. **Summary construction:** select the highest-scoring sentences and restore their source order.
    7. **Evaluation:** optionally compare the result with a reference summary using ROUGE overlap metrics.
    """)
    st.subheader("Scope and limitations")
    st.markdown("""
    - This is **extractive** summarization: it selects source sentences and does not paraphrase them.
    - Sentence splitting is lightweight and may not correctly handle every abbreviation or language.
    - ROUGE scores measure overlap with a reference summary, not complete semantic quality.
    - Results should be tested on multiple documents; a single successful example is not enough to establish performance.
    """)
    st.subheader("Suggested project objective")
    st.write("To design and implement a local NLP application that summarizes text using a TextRank-style graph-ranking method and evaluates summary overlap with reference text.")
