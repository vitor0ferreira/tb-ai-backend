# Use uma imagem base oficial do Python. A versão 'slim' é mais leve.
FROM python:3.9-slim

# Defina o diretório de trabalho dentro do container
WORKDIR /code

# Copie o arquivo de dependências primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie todo o resto do seu código para o diretório de trabalho
COPY . .

# Exponha a porta em que a aplicação vai rodar. 7860 é um padrão comum no HF.
EXPOSE 7860

# O comando para iniciar a aplicação quando o container for executado.
# Usamos Gunicorn para produção, escutando em todas as interfaces na porta 7860.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:app"]