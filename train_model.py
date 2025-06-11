import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- Parâmetros de Configuração ---
IMG_WIDTH, IMG_HEIGHT = 224, 224 # Tamanho que a MobileNetV2 usa
BATCH_SIZE = 32
TRAIN_DATA_DIR = 'dataset/train'
VALIDATION_DATA_DIR = 'dataset/validation'
EPOCHS = 10 # Número de vezes que o modelo vai rodar todo o dataset

def build_model():

    # Carrega o modelo base MobileNetV2, pré-treinado no dataset ImageNet
    # include_top=False: remove a camada de classificação final original do modelo.
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # Congela as camadas do modelo base para que não sejam treinadas novamente.
    for layer in base_model.layers:
        layer.trainable = False

    # Adiciona nossas próprias camadas de classificação no topo do modelo base
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Reduz a dimensionalidade
    x = Dense(1024, activation='relu')(x) # Camada densa para aprender padrões complexos
    x = Dropout(0.5)(x) # Dropout para evitar overfitting (memorização)
    # Camada de saída com 1 neurônio e ativação sigmoide,
    # ideal para classificação binária (tuberculose ou normal).
    # O resultado será um valor entre 0 e 1 (a probabilidade).
    predictions = Dense(1, activation='sigmoid')(x)

    # Cria o modelo final
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def main():
    """
    Função principal que executa a preparação dos dados, compilação e treinamento do modelo.
    """
    # --- Preparação dos Dados ---

    # ImageDataGenerator aplica transformações às imagens em tempo real (Data Augmentation)
    # para aumentar a variedade do dataset e evitar overfitting.
    train_datagen = ImageDataGenerator(
        rescale=1./255, # Normaliza os pixels para o intervalo [0, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255) # Normalizamos dados de validação

    # Cria geradores de dados que leem as imagens
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary' # (normal/tuberculosis)
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # --- Construção e Compilação do Modelo ---
    model = build_model()

    # Compila o modelo, definindo o otimizador, a função de perda e as métricas
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    # --- Treinamento do Modelo ---
    print("Iniciando o treinamento do modelo...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # --- Salvando o Modelo Treinado ---
    # O modelo salvo conterá a arquitetura, os pesos e a configuração do treinamento.
    model.save('tuberculosis_detector.h5')
    print("Modelo treinado e salvo como 'tuberculosis_detector.h5'")


if __name__ == '__main__':
    main()