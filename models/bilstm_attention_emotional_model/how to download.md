# 📦 Trained Model Download — BiLSTM + Attention

The trained model file:

bilstm_attention_emotions.keras

is stored externally on **Google Drive** due to GitHub's 100MB file size limitation.

# ⬇ Download Link

Download the model from this public Drive link:

👉 https://drive.google.com/file/d/12s9vnycuhP-XZgWlKRJrYdgoi3rRQrUE/view?usp=sharing

# 📁 After Download — Folder Placement

After downloading, place the file exactly here:

bilstm_attention_emotional_model/
└── bilstm_attention_emotions.keras

Your final bilstm_attention_emotional_model folder must look like this:

bilstm_attention_emotional_model/
├── tokenizer.pkl
└── bilstm_attention_emotions.keras

# ✅ Verification

To verify that the model is correctly placed, run this in Python:

```python
import os
print(os.listdir("bilstm_attention_emotional_model"))
```

You should see:

['tokenizer.pkl', 'bilstm_attention_emotions.keras']

# ⚠ Important Notes

- This is a **FULL Keras model file**.
- It includes:
  - Model architecture
  - Trained parameters
  - Custom Attention layer configuration
- You must load it using:

```python
tf.keras.models.load_model(
"models/bilstm_attention_emotional_model/bilstm_attention_emotions.keras",
custom_objects={"Attention": Attention}
)
```

- Do NOT rename the file unless you also update:
  - prediction.ipynb
  - streamlit_app.py

# ✅ Usage

Once the model is placed correctly, you can directly run:

```bash
streamlit run streamlit_app.py
```

No retraining is required.

# 🔒 Hosting Reason

This model is hosted on Google Drive because:

- File size exceeds GitHub’s hard 100MB limit
- This avoids bloating the repository
- Enables fast direct download for all users

# ✅ Final Note

If the app fails to start, 99% of the time the reason is:

- The model file is missing
- The file name is incorrect
- The file is placed in the wrong directory

Always confirm the exact path before debugging anything else.
