# Cross-Country Multi-Label Dialectal Arabic Identification

## Authors
- **Ali Mekky**  
  MBZUAI / MSc NLP  
  [ali.mekky@mbzuai.ac.ae](mailto:ali.mekky@mbzuai.ac.ae)

- **Lara Hassan**  
  MBZUAI / MSc CS  
  [lara.hassan@mbzuai.ac.ae](mailto:lara.hassan@mbzuai.ac.ae)

- **Mohamed ElZeftawy**  
  MBZUAI / MSc CS  
  [mohamed.elzeftawy@mbzuai.ac.ae](mailto:mohamed.elzeftawy@mbzuai.ac.ae)

---

## Abstract
This project focuses on the Multi-label country-level Dialect Identification (ML-DID) subtask for NADI2024 at the ArabicNLP conference. We address the challenges of accurately identifying Arabic dialects by introducing a novel MLDID dataset, using prompt engineering and automated augmentation via n-binary classifiers. Benchmarking diverse machine-learning models, we achieved a macro F1-score of 65.49%, outperforming the top-performing team in the NADI 2024 shared task.

---

## Introduction
Arabic is spoken by over 420 million people across 28+ nations, encompassing diverse dialects. These dialects present significant challenges for Natural Language Processing (NLP), particularly in informal communication contexts. Traditional single-label Arabic Dialect Identification (ADI) approaches often fail to address overlapping dialectal features, motivating the shift to a multi-label framework.

---

## Dataset
We utilized datasets from NADI2020, NADI2021, and NADI2023, converting single-label annotations into multi-label formats. The dataset includes 31,760 records with balanced representations across 18 Arabic dialects. Preprocessing involved text normalization, cleaning, and augmentation using tools like NLTK, Camel Tools, and PyArabic.

---

## Methodology
### Dataset Creation by Pseudo-labeling 
Three approaches were employed for dataset creation:
1. **Logistic Regression:** Used 18 binary classifiers to assign multiple labels.
2. **MARBERT Fine-Tuning:** Leveraged the transformer-based model pre-trained on dialectal Arabic tweets.
3. **GPT-4 Prompt Engineering:** Generated pseudo-labels using carefully crafted prompts.

### Multi-Label Classification
We fine-tuned MARBERT and CAMeLBERT models, implementing curriculum-based training to handle dataset imbalances and improve multi-label classification performance.


---

## Experiments and Evaluation
Experiments were conducted using macro F1-score, precision, recall, and accuracy metrics. Curriculum-based training proved effective, achieving superior performance over traditional methods. Key results include:
- **Macro F1-Score:** 65.49%
- **Significant improvement over NADI2024 benchmarks.**

---

## Challenges and Future Work
Key challenges include:
- Distinguishing closely related dialects.
- Addressing noise from geographic metadata in dataset annotations.

Future directions:
- Refining annotation methodologies.
- Enhancing data augmentation techniques.
- Scaling models for broader linguistic coverage.

---

## Limitations
- Geographic-based annotations introduce noise.
- Multi-label dataset conversion was not manual, potentially impacting accuracy.
- Evaluation was limited to the development set, as the test set was unavailable.


