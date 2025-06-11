import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf

# --- Configuração da Aplicação Flask ---
app = Flask(__name__)
# Habilita o CORS para permitir requisições do front-end da Vercel
CORS(app) 

# --- Carregamento do Modelo ---
MODEL_PATH = 'tuberculosis_detector.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modelo {MODEL_PATH} carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    model = None

# --- Parâmetros ---
IMG_WIDTH, IMG_HEIGHT = 224, 224

def preprocess_image(image_file):
    """
    Processa o arquivo de imagem para o formato que o modelo espera.
    1. Abre a imagem.
    2. Converte para RGB (caso seja RGBA ou P).
    3. Redimensiona para 224x224 pixels.
    4. Converte para um array numpy.
    5. Normaliza os pixels para o intervalo [0, 1].
    6. Adiciona uma dimensão extra para o batch.
    """
    img = Image.open(image_file).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Cria o formato (1, 224, 224, 3)
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    """
    Recebe uma imagem, a processa, faz a predição e retorna o resultado.
    """
    if model is None:
        return jsonify({'error': 'Modelo não está carregado'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nome de arquivo inválido'}), 400

    try:
        processed_image = preprocess_image(file)

        prediction = model.predict(processed_image)
        
        # O resultado do modelo é um array, pegamos o primeiro (e único) valor.
        # Este valor é a probabilidade da imagem ser da classe 'tuberculosis'.
        probability = float(prediction[0][0])
        
        # Retorna o resultado em formato JSON
        return jsonify({
            'probability_tuberculosis': probability,
            'class_name': 'tuberculosis' if probability > 0.5 else 'normal'
        })

    except Exception as e:
        return jsonify({'error': f'Erro ao processar a imagem: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)