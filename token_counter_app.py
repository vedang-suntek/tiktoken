import os
import tempfile
from flask import Flask, request, jsonify, render_template
import tiktoken
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Available models and their corresponding encoding
MODELS = {
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "davinci": "p50k_base",
}

@app.route('/')
def index():
    return render_template('index.html', models=list(MODELS.keys()))

@app.route('/count-tokens', methods=['POST'])
def count_tokens():
    data = request.get_json()
    text = data.get('text', '')
    model = data.get('model', 'gpt-4')
    
    encoding_name = MODELS.get(model, 'cl100k_base')
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to get_encoding if model is not directly supported
        encoding = tiktoken.get_encoding(encoding_name)
        
    tokens = encoding.encode(text)
    
    return jsonify({
        'token_count': len(tokens),
        'character_count': len(text),
        'tokens': tokens
    })

@app.route('/upload-file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    model = request.form.get('model', 'gpt-4')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    encoding_name = MODELS.get(model, 'cl100k_base')
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to get_encoding if model is not directly supported
        encoding = tiktoken.get_encoding(encoding_name)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = encoding.encode(text)
        
        # Cleanup the temporary file
        os.unlink(filepath)
        
        return jsonify({
            'token_count': len(tokens),
            'character_count': len(text),
            'filename': filename
        })
    except Exception as e:
        # Cleanup the temporary file in case of error
        if os.path.exists(filepath):
            os.unlink(filepath)
        return jsonify({'error': str(e)}), 500

# Currently images are handled simply by their approximate token usage in vision models
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Get image dimensions to approximate token usage
        # For simplicity, using a fixed token count for demonstration
        # In production, you'd use proper image analysis
        approx_tokens = 85  # Default low-res image token count
        
        # Cleanup the temporary file
        os.unlink(filepath)
        
        return jsonify({
            'token_count': approx_tokens,
            'filename': filename,
            'note': 'Image token count is approximate and based on GPT-4 Vision specifications'
        })
    except Exception as e:
        # Cleanup the temporary file in case of error
        if os.path.exists(filepath):
            os.unlink(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000) 