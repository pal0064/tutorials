from python:3.10.6-slim-buster
RUN apt-get update -y
RUN apt-get -y install gcc python3-dev
RUN pip install --upgrade pip
WORKDIR app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ./ /app/
EXPOSE 8888
ENTRYPOINT ["jupyter","notebook", "--ip" ,"0.0.0.0", "--no-browser", "--allow-root","--NotebookApp.token=''","--NotebookApp.password=''"]