import os
import tempfile
from flask import Flask, request, jsonify, render_template
import tiktoken
from werkzeug.utils import secure_filename
import math
import json
import traceback

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
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        text = data.get('text', '')
        model = data.get('model', 'gpt-4')
        
        if model in MODELS:
            encoding_name = MODELS[model]
            encoding = tiktoken.get_encoding(encoding_name)
            tokens = encoding.encode(text)
            return jsonify({
                'token_count': len(tokens),
                'character_count': len(text),
                'word_count': len(text.split())
            })
        else:
            return jsonify({'error': 'Invalid model selected'}), 400
    except Exception as e:
        app.logger.error(f"Error in count_tokens: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/count-file-tokens', methods=['POST'])
def count_file_tokens():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        model = request.form.get('model', 'gpt-4')
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if model in MODELS:
                    encoding_name = MODELS[model]
                    encoding = tiktoken.get_encoding(encoding_name)
                    tokens = encoding.encode(content)
                    result = {
                        'token_count': len(tokens),
                        'character_count': len(content),
                        'word_count': len(content.split()),
                        'filename': filename
                    }
                    return jsonify(result)
                else:
                    return jsonify({'error': 'Invalid model selected'}), 400
                    
            except UnicodeDecodeError:
                # File might be binary/image
                return jsonify({'error': 'File appears to be binary or non-text. Please upload a text file.'}), 400
            finally:
                # Clean up the temp file
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        return jsonify({'error': 'Failed to process file'}), 500
    except Exception as e:
        app.logger.error(f"Error in count_file_tokens: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/count-image-tokens', methods=['POST'])
def count_image_tokens():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part'}), 400
        
        image = request.files['image']
        model = request.form.get('model', 'gpt-4-vision-preview')
        detail = request.form.get('detail', 'high')
        
        if image.filename == '':
            return jsonify({'error': 'No selected image'}), 400
        
        if image:
            filename = secure_filename(image.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(file_path)
            
            try:
                # Get image dimensions using pillow
                from PIL import Image as PILImage
                with PILImage.open(file_path) as img:
                    width, height = img.size
                
                # Calculate tokens based on the official GPT-4 Vision token calculation algorithm
                token_count = calculate_vision_tokens(width, height, detail)
                
                # Calculate tiles for response
                tiles_width = math.ceil(width / 512)
                tiles_height = math.ceil(height / 512)
                
                result = {
                    'token_count': token_count,
                    'width': width,
                    'height': height,
                    'detail': detail,
                    'filename': filename,
                    'tiles_width': tiles_width,
                    'tiles_height': tiles_height,
                    'total_tiles': tiles_width * tiles_height
                }
                
                # Log the result for debugging
                app.logger.info(f"Image processing result: {json.dumps(result)}")
                
                return jsonify(result)
                    
            except Exception as e:
                app.logger.error(f"Error processing image: {str(e)}")
                app.logger.error(traceback.format_exc())
                return jsonify({'error': f'Failed to process image: {str(e)}'}), 400
            finally:
                # Clean up the temp file
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        return jsonify({'error': 'Failed to process image'}), 500
    except Exception as e:
        app.logger.error(f"Error in count_image_tokens: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

def calculate_vision_tokens(width, height, detail='high'):
    """
    Calculate token usage for images based on the official OpenAI documentation.
    
    In high detail mode:
    1. Image is resized to max 2048x2048
    2. Scaled such that the shortest side is 768px (if original dimensions exceed 768px)
    3. Divided into 512x512 tiles
    4. Each tile costs 170 tokens + base cost of 85 tokens
    
    In low detail mode: fixed 85 tokens
    """
    if detail == 'low':
        return 85
    
    # Scale down to fit within a 2048 x 2048 square if necessary
    if width > 2048 or height > 2048:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            width = 2048
            height = int(2048 / aspect_ratio)
        else:
            height = 2048
            width = int(2048 * aspect_ratio)
    
    # Resize such that the shortest side is 768px if the original dimensions exceed 768px
    min_size = 768
    aspect_ratio = width / height
    if width > min_size and height > min_size:
        if aspect_ratio > 1:
            height = min_size
            width = int(min_size * aspect_ratio)
        else:
            width = min_size
            height = int(min_size / aspect_ratio)
    
    # Calculate tiles
    tiles_width = math.ceil(width / 512)
    tiles_height = math.ceil(height / 512)
    
    # Calculate total tokens (85 base + 170 per tile)
    total_tokens = 85 + 170 * (tiles_width * tiles_height)
    
    return total_tokens

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 