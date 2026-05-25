# ============================================================
# Dockerfile pentru Wine Quality API
# Curs: Aplicatii Cloud - Master Stiinta Datelor, UTM
# ============================================================

# Pasul 1: Imaginea de baza
# Folosim Python 3.11 in varianta "slim" (Debian minimal, ~150 MB).
# Slim are doar ce e necesar pentru a rula Python, fara extra OS tools.
FROM python:3.11-slim

# Pasul 2: Metadate (opțional dar profesional)
LABEL maintainer="Popa Olesea <olesea.popa@example.md>"
LABEL description="Wine Quality API - FastAPI + scikit-learn"
LABEL version="1.0.0"

# Pasul 3: Setam directorul de lucru in container
# Toate comenzile urmatoare se executa relativ la /app
WORKDIR /app

# Pasul 4: Setam variabile de mediu pentru Python
# PYTHONDONTWRITEBYTECODE: nu creeaza fisiere .pyc (economisim spatiu)
# PYTHONUNBUFFERED: output Python apare imediat in logs (esential pentru debugging)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Pasul 5: Copiem MAI INTAI requirements.txt si instalam dependentele
# Acest pas separat de copierea codului permite Docker sa REFOLOSEASCA layer-ul
# de dependente daca doar codul s-a schimbat (build-uri ulterioare mult mai rapide)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pasul 6: Copiem codul aplicatiei si modelul antrenat
# Le copiem dupa instalarea dependentelor (din motivul de mai sus)
COPY app.py .
COPY model.pkl .
COPY features.pkl .

# Pasul 7: Documentam portul pe care aplicatia asculta
# EXPOSE nu deschide efectiv portul - doar documenteaza intentia
# (deschiderea efectiva se face cu -p la docker run)
EXPOSE 8000

# Pasul 8: Comanda de pornire a containerului
# Folosim forma "exec" (cu lista JSON) - recomandata pentru semnal handling corect
# host 0.0.0.0 = asculta pe toate interfetele (esential in container!)
# port 8000 = portul standard al aplicatiei noastre
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
