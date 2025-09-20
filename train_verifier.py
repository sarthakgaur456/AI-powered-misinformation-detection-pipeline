import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

PAIRS_CSV = 'claim_evidence_pairs.csv'
MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
OUT_DIR = 'models/verifier_crossencoder'
BATCH_SIZE = 16
EPOCHS = 2
RANDOM_STATE = 42

label2id = {'supports': 1.0, 'refutes': 0, 'uncertain': -1.0}

df = pd.read_csv(PAIRS_CSV)
#keep only those rows with allowed labels
df = df[df['label'].isin(list(label2id.keys()))].reset_index(drop = True)

print('Total pairs:', len(df))

#splitting
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=RANDOM_STATE)
print('Train:', len(train_df), 'Val:', len(val_df))
train_df.to_csv("pairs_train.csv", index=False)
val_df.to_csv("pairs_val.csv", index=False)

train_examples = []
for _, r in train_df.iterrows():
    txts = [str(r['claim']), str(r['evidence'])]
    score = float(label2id[r['label']])
    train_examples.append(InputExample(texts=txts, label=score))

val_examples = []
for _, r in val_df.iterrows():
    txts = [str(r['claim']), str(r['evidence'])]
    score = float(label2id[r['label']])
    val_examples.append(InputExample(texts=txts, label=score))

#create the cross-encoder:
print('Loading cross-encoder:', MODEL_NAME)
model = CrossEncoder(MODEL_NAME, num_labels= 1)

train_dataloader = DataLoader(train_examples, shuffle= True, batch_size=BATCH_SIZE)

print('Training for', EPOCHS, 'epochs')
model.fit(train_dataloader, epochs = EPOCHS, show_progress_bar=True, output_path=OUT_DIR)

model.save(OUT_DIR)
print('Model saved to:', OUT_DIR)


