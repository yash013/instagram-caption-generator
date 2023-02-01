from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    keywords = request.form['keywords']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(app.config['UPLOAD_FOLDER'] + filename)
        # Generate a list of potential captions
        keywords = word_tokenize(keywords)
        captions = []
        for keyword in keywords:
            synonyms = wordnet.synsets(keyword)
            for syn in synonyms:
                captions.append(syn.definition())
        
        # Determine the sentiment of each caption
        sia = SentimentIntensityAnalyzer()
        scores = []
        for caption in captions:
            scores.append(sia.polarity_scores(caption))
        
        # Determine the most appropriate caption
        sia = SentimentIntensityAnalyzer()
        top_captions = sorted(captions, key=lambda x: sia.polarity_scores(x)['compound'], reverse=True)[:3]

        message = f"Your file has been uploaded and your keywords are: {keywords}"  
        return json.dumps({"message": message, "captions":captions, "top_captions":top_captions},), 200, {'ContentType':'application/json'}
    else:
        message = "Invalid file type. Please upload a jpg, jpeg, png, or gif."
        return json.dumps({"message": message}), 400, {'ContentType':'application/json'}
        

@app.route('/')
def index():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
