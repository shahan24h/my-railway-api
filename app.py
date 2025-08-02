from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification

model_path = "mednarr10k_model_02"

# Correct tokenizer/model pair
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    response = jsonify({"prediction": prediction}); response.headers["Content-Type"] = "application/json"; return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway will set PORT
    app.run(host="0.0.0.0", port=port)


