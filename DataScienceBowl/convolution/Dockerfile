FROM kaixhin/theano

MAINTAINER hursh.prasad.14@ucl.ac.uk

# linux dependency for matlibplot
RUN apt-get update && apt-get install --force-yes -y \
    libpng-dev \
    zlib1g-dev \
    libfreetype6-dev \
    libxft-dev

# upgrade packages
RUN pip freeze > requirements.txt && pip install -r requirements.txt --upgrade

# install libs for python pyCharm Interpreter
RUN pip install pydicom

# plotting libraries
RUN pip install pylab Image

# To build image
# docker build -t aml-datascience-bowl .
# change pyCharm Interpreter

# This will allow printing images
# check output layers of CNN
