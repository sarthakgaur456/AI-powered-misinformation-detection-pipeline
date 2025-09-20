import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sentence_transformers import CrossEncoder

VAL_CSV = 'pairs_val.csv'
MODEL_DIR = 'models/verifier_crossencoder'

def score_to_label(score, t_refute = -0.25, t_support = 0.25):
    if score <= t_refute:
        return 'refutes'
    if score >= t_support:
        return 'supports'
    return 'uncertain'


test_df = pd.read_csv(VAL_CSV)
print('Evaluating on', len(test_df), 'pairs')

model = CrossEncoder(MODEL_DIR)

pairs = [[str(r['claim']), str(r['evidence'])] for _, r in test_df.iterrows()]
scores = model.predict(pairs).squeeze()

pred_labels = [score_to_label(float(s)) for s in scores]

print('\nClassification report:')
print(classification_report(test_df['label'].tolist(), pred_labels, digits=4))

out = test_df.copy()
out['pred_score'] = scores
out['pred_label'] = pred_labels
os.makedirs('models', exist_ok=True)
out.to_csv('models/verifier_eval.csv', index = False)
print("\nPredictions saved to models/verifier_eval.csv")