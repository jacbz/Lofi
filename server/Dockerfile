FROM python:3.12-slim

WORKDIR /app

COPY server/requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./server server
COPY ./model model
COPY ./checkpoints checkpoints

CMD [ "python3", "-m" , "flask", "--app=server/app.py", "run", "--host=0.0.0.0"]
