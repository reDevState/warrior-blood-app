FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python3 -c "open('/usr/local/lib/python3.12/site-packages/numba/misc/coverage_support.py','w').write('\"\"\"Numba coverage stub.\"\"\"\nfrom __future__ import annotations\n_coverage_available=False\ndef get_registered_loc_notify():\n    return None\n')" || true

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
