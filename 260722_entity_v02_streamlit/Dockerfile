FROM python:3.12-slim

ENV HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py timeline_data.py training_mixtures.py seed_rows.json reference_rows.json ground_truth_shot_statistics.json entity_duration_statistics.json training_token_mixtures.json gemini_timeline_data.json ./

RUN useradd --no-create-home --uid 10001 dashboard
USER 10001

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
