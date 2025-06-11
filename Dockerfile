# Usa uma imagem base oficial do Python, 'slim' é mais leve.
FROM python:3.9-slim

# Define diretório de trabalho dentro do container
WORKDIR /code

# Copia o arquivo de dependências primeiro
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o resto do código para o diretório de trabalho
COPY . .

# Exponhe a porta em que a aplicação vai rodar. 7860 é padrão no HF.
EXPOSE 7860

# O comando para iniciar a aplicação quando o container for executado.
# Usa Gunicorn para produção, escutando em todas as interfaces na porta 7860.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:app"]