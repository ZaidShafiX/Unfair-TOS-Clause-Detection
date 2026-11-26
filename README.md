# Unfair Terms of Service Clause Detection

**Live demo:** https://unfair-tos-clausedetection-evufwrzfgpfzfgbjffmark.streamlit.app/

Course: AIC354 Machine Learning Fundamentals, COMSATS University Islamabad, Lahore. Assignment 4: End-to-End ML Systems Development using Unstructured Data.
Author: Muhammad Zaid Shafi (FA24-BSE-097)

## Problem
Classify a single Terms of Service clause as **Fair** or **Unfair** (e.g. unilateral termination, forced arbitration, liability waivers).
Imbalanced binary classification: about 11% Unfair (8.04 : 1 after de-duplication).

## Data
LexGLUE `unfair_tos` (`coastalcph/lex_glue`): 9,414 clauses from 50 contracts, 9,179 after de-duplication.
Stratified 80/20 split (7,343 train / 1,836 test, 203 Unfair test clauses). The test set is used once, for the final report.

## Method
- Cleaning keeps legal negations and modals ("not", "may", "without"), expands contractions, removes duplicates.
- Tier 1: TF-IDF (1-2 grams, 5,000 features) with 6 classical algorithms, with and without class weights.
- Tier 2/3: fine-tuned DistilBERT and Legal-BERT with a class-weighted loss.
- Models were selected without the test set: classical by 5-fold CV, transformers by validation F1.

## Results (test set, Unfair class)
| Model | F1 | Recall | Precision | PR-AUC |
|---|---|---|---|---|
| Baseline (always Fair) | 0.000 | 0.000 | 0.000 | - |
| Linear SVM (Balanced), deployed | 0.736 | 0.764 | 0.711 | 0.796 |
| Random Forest (Balanced) | 0.770 | 0.734 | 0.810 | 0.839 |
| DistilBERT | 0.792 | 0.818 | 0.769 | 0.849 |
| Legal-BERT | 0.812 | 0.818 | 0.806 | 0.877 |

Notes: the always-Fair baseline reaches 88.9% accuracy, which is why F1/recall/PR-AUC are used.
The top three classical models are tied on CV (0.7211, 0.7185, 0.7183, std 0.024-0.043).
DistilBERT vs Legal-BERT differs by about 0.02 F1 from a single run, which is within plausible noise.

## Deployment trade-off
The live app ships the Linear SVM: 40 KB model + 201 KB vectorizer, median 3.1 ms per clause on CPU.
Legal-BERT is the most accurate model but needs PyTorch and about 440 MB of weights, with a median of 115 ms per clause on the same CPU (measured on the author's laptop).

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Files: `app.py`, `best_model.pkl`, `tfidf_vectorizer.pkl`, `requirements.txt`.
