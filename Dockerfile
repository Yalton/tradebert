#Blog (Dockerfile)
FROM python:3.9

WORKDIR $(pwd):/usr/src/app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

CMD [ "python", "./interface.py"]