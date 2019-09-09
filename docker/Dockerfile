FROM ubuntu:16.04

MAINTAINER DIDARUL ""

RUN apt -y update &&\
    apt -y install python3 python3-pip

RUN python3 -m pip install --upgrade pip

ADD ./python_requirements.txt /
RUN python3 -m pip install -r python_requirements.txt
RUN [ "python3", "-c", "import nltk; nltk.download('brown')" ]
RUN [ "python3", "-c", "import nltk; nltk.download('punkt')" ]

# ADD ../app /
ENTRYPOINT ["bash","/app/start_flask.sh"]


