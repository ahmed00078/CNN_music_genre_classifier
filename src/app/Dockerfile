# Utiliser une image de base Python
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de l'application dans le conteneur
COPY requirements.txt requirements.txt

# Installer les dépendances
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Copier tout le contenu du projet dans le conteneur
COPY . .

# Exposer le port Streamlit (8501 par défaut)
EXPOSE 8501

# Lancer l'application Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]