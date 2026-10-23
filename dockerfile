FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libgeos-dev libproj-dev binutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN DJANGO_SECRET_KEY=build-only-dummy-key python manage.py collectstatic --noinput

CMD sh -c "python manage.py migrate --noinput && gunicorn taskflow.wsgi:application --bind 0.0.0.0:${PORT:-10000} --workers 1 --timeout 120"
