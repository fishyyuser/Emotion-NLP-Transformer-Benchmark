# 📦 Trained Model Download — BERT Emotion Classifier (Transformer)

The fine-tuned BERT model folder:

bert_emotion_model/

is stored externally on Google Drive as a ZIP archive due to GitHub's 100MB file size limitation.

# ⬇ Download Link

Download the model zip from this public Drive link:

https://drive.google.com/file/d/1FLqvNg3yDpF-D3aLxn3y0mjkod8YlJwp/view?usp=sharing

This zip contains the full Hugging Face model directory.

# 📁 After Download — Folder Placement

After downloading, extract the zip file.

You must place the extracted folder exactly here:

bert_emotion_model/

Your final bert_emotion_model folder must look like this:

```bash
bert_emotion_model/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

# ✅ Verification

To verify that the model is correctly placed, run this in Python:

```bash
import os
print(os.listdir("bert_emotion_model"))
```

You should see something like:

```bash
['config.json',
'model.safetensors',
 'tokenizer.json',
 'tokenizer_config.json',
 'vocab.txt']
```

# ⚠ Important Notes

- This is a FULL Hugging Face Transformers model directory.
- It includes:
  - Model configuration
  - Trained BERT parameters
  - Full tokenizer files
- You must load it using:

```bash
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert_emotion_model")
model = AutoModelForSequenceClassification.from_pretrained("bert_emotion_model")
model.eval()
```

- Do NOT rename the folder unless you also update:
  - streamlit_app.py
  - bert evaluation / prediction notebooks

# ✅ Usage

Once the model is placed correctly, you can directly run:

```bash
streamlit run streamlit_app.py
```

No retraining is required.

# 🔒 Hosting Reason

This model is hosted on Google Drive because:

- Folder size exceeds GitHub’s 100MB hard limit
- This avoids bloating the repository
- Enables fast direct download for all users

# ✅ Final Note

If the app fails to load the BERT model, 99% of the time the reason is:

- The model folder is missing
- The folder name is incorrect
- The folder is placed in the wrong directory
- One or more files failed to extract properly

Always confirm the exact path before debugging anything else.
