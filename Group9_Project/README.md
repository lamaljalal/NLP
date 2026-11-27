
# Group 9 — SMS Spam Detection

## Project Overview
This project investigates SMS spam detection using three NLP approaches and compares their classification performance against training and inference costs.

**Course:** ARTI 452 — Natural Language Processing  
**Group:** 9  
**Academic Year:** 2026–2027

## Research Question
Do the classification performance gains of fine-tuned DistilBERT justify its additional training and inference costs relative to frozen MiniLM sentence embeddings and a TF-IDF baseline?

## Models
1. TF-IDF + Logistic Regression
2. Frozen all-MiniLM-L6-v2 embeddings + Logistic Regression
3. Fine-tuned DistilBERT

## Dataset
UCI SMS Spam Collection:
- Original messages: 5,574
- After exact duplicate removal: 5,158
- Training: 3,610
- Development: 774
- Testing: 774

## Results

| Model | Accuracy | Macro F1 | Spam Recall |
|---|---:|---:|---:|
| TF-IDF | 98.45% | 96.46% | 94.79% |
| Frozen MiniLM | 98.45% | 96.53% | 96.88% |
| Fine-tuned DistilBERT | 99.22% | 98.20% | 95.83% |

## Repository Files
- `SMSSpamDetection.ipynb` — Main notebook containing the implementation and evaluation.
- `results_consistent_timing.csv` — Model evaluation and timing results.
- `timing_protocol_hardware.json` — Hardware details and timing methodology.
- `tuning_costs.csv` — Hyperparameter tuning costs.

## How to Run
1. Open `SMSSpamDetection.ipynb` in Google Colab.
2. Enable a GPU runtime, preferably a Tesla T4.
3. Run the notebook cells sequentially, starting from setup and data preparation.
4. Follow the notebook's data-loading, training, evaluation, and model-saving steps.
5. Use the **Try it** section to enter a new SMS message and obtain a Spam or Not Spam prediction.

**Note:** Running on different hardware or software versions may produce different timing results.

## Hardware and Timing
The reported experiments used a Tesla T4 GPU. Training times include relevant feature extraction or embedding steps. Inference was measured using five repeated runs after warm-up, reporting the median time.

## Team
- Ghala Ahmed Saeed Alghamdi
- Lama Tariq Saad Aljalal
- Khawlah Majed Ibrahim Abaalkhail
- Reemas Tariq Abdullatef Aleid

## Dataset Reference
UCI SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Supplementary notebooks
- `Data_Preprocessing.ipynb`: data preparation and split verification.
- `SMS_Classification_Testing.ipynb`: independent verification run and new SMS tests. Its training times differ from the primary run (0.2002, 4.0257, 67.2037 s); inference times are 0.0324, 0.8097, 1.7368 s. Do not mix runs.
- `Robustness_Testing-2.ipynb`: 20 synthetic messages, six variants, 120 cases and 360 predictions. Historical outputs use the verification notebook's saved models. Run its save section first; keep the same Colab runtime or transfer the entire `group9_saved_models` folder to the new runtime and set `MODEL_DIR`. No personal Drive path is required.

The primary benchmark is `SMSSpamDetection.ipynb` and the three accompanying CSV/JSON files. Supplementary results are not held-out test scores. Saved outputs are historical; rerunning may change timing.

