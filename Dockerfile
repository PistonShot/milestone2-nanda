FROM  python:3.10.11-slim-buster

WORKDIR /python-docker

COPY . .
RUN pip3 install -r requirements.txt

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]