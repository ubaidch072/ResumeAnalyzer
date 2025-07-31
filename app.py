
from flask import Flask, render_template, request, send_file
import os
import PyPDF2
import pandas as pd
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the ML model and vectorizer
with open(os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)
with open(os.path.join(BASE_DIR, 'models', 'role_classifier.pkl'), 'rb') as f:
    classifier = pickle.load(f)

results_df = pd.DataFrame(columns=['filename', 'prediction'])

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return ' '.join(page.extract_text() or '' for page in reader.pages)
    except Exception:
        return ''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_single', methods=['POST'])
def analyze_single():
    name = request.form['name']
    job_description = request.form['job_description']
    resume_file = request.files['resume']

    filename = secure_filename(resume_file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    resume_file.save(file_path)

    resume_text = extract_text_from_pdf(file_path)
    combined_text = job_description + " " + resume_text
    X = vectorizer.transform([combined_text])
    prediction = classifier.predict(X)[0]

    result = [{'filename': f"{name} ({filename})", 'prediction': prediction}]
    return render_template('result.html', results=result)

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    global results_df
    skills = request.form['skills']
    resume_files = request.files.getlist('resumes')
    results = []
    results_df = pd.DataFrame(columns=['filename', 'prediction'])

    for file in resume_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        text = extract_text_from_pdf(file_path)
        combined_text = skills + " " + text
        if text.strip():
            X = vectorizer.transform([combined_text])
            prediction = classifier.predict(X)[0]
        else:
            prediction = "Could not extract text"

        results.append({'filename': filename, 'prediction': prediction})
        results_df = pd.concat([results_df, pd.DataFrame([{'filename': filename, 'prediction': prediction}])])

    return render_template('result.html', results=results)

@app.route('/download_csv')
def download_csv():
    csv_path = os.path.join(UPLOAD_FOLDER, 'results.csv')
    results_df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
