from flask import Flask, request, jsonify
import spacy
from transformers import pipeline

#python C:\Users\emree\OneDrive\Masaüstü\MedicalReport\MedicalReportPython\medicalReportPython.py
nlp = spacy.load("en_core_web_sm")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

app = Flask(__name__)

@app.route('/api/predict/spacy', methods=['POST'])
def predict_spacy():
    data = request.json
    if not data or 'text' not in data:  
        return jsonify({'error': 'Text field is required'}), 400

    text = data['text']  
    doc = nlp(text)  

    result = []
    for token in doc:
        entity_type = token.ent_type_ if token.ent_type_ else "O"  
        result.append({"text": token.text, "entityType": entity_type})

    return jsonify({'body': {'tokens': result}}) 

@app.route('/api/predict/bert', methods=['POST'])
def predict_bert():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Text field is required'}), 400

    text = data['text']
    try:
        results = ner_pipeline(text)

        entities = []
        temp_entity = {"text": "", "entityType": ""}
        for entity in results:
            word = entity['word']
            entity_type = entity['entity']
            
            if entity_type != temp_entity["entityType"] or (not word.startswith("##") and temp_entity["text"]):
                if temp_entity["text"]:
                    entities.append(temp_entity)
                temp_entity = {"text": word.lstrip("##"), "entityType": entity_type}
            else:
                if word.startswith("##"):  
                    temp_entity["text"] += word[2:]
                else:
                    temp_entity["text"] += f" {word}"
        
        if temp_entity["text"]:
            entities.append(temp_entity)

        return jsonify({'body': {'tokens': entities}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
