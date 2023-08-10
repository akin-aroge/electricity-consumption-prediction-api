FROM python:3.10.12-bullseye

RUN mkdir -p /app
COPY data/ /app/data
COPY feature_store/ /app/feature_store
COPY models/ /app/models
COPY src /app/src
COPY streamlit_utils app/streamlit_utils
COPY app.py /app/app.py
COPY Makefile /app/Makefile
COPY pip_reqs.txt /app/pip_reqs.txt

WORKDIR /app
# RUN make install
RUN pip install --no-cache-dir --upgrade -r pip_reqs.txt

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
