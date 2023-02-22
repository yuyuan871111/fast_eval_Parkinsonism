# The build-stage image:
FROM continuumio/miniconda3 AS build

# Update conda and install conda-pack:
RUN conda update conda && conda install -c conda-forge conda-pack

# Set conda mediapipe environment (time consuming step)
COPY environment.yml .
RUN conda env create -f environment.yml

# Use conda-pack to create a standalone enviornment in /venv:
RUN conda-pack -n mediapipe -o /tmp/env.tar \
    && mkdir /venv && cd /venv && tar xf /tmp/env.tar \
    && rm /tmp/env.tar

# We've put venv in same path it'll be in final image, so now fix up paths:
RUN /venv/bin/conda-unpack

##########################
# Create the base OS image (Ruby)
FROM ruby:3.1.2 AS base

# Define build arguments
ARG USER_ID=1001
ARG GROUP_ID=1001

# SET ENV/ARG/WKDIR PATH
ARG SETUSER="myuser"
ARG APP_ROOT="/home/${SETUSER}/local/app"
ARG CONDA_PKG="/home/${SETUSER}/local/conda_pkg"
RUN addgroup --gid ${GROUP_ID} ${SETUSER}
RUN adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} ${SETUSER}
RUN mkdir -p $APP_ROOT && mkdir -p CONDA_PKG
WORKDIR $APP_ROOT

# Copy /venv from the previous stage:
COPY --from=build /venv ${CONDA_PKG}

# Install ffmepg, wget, sqlite3, zip, python3-opencv
RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends ffmpeg \
    && apt-get install -y wget \
    && apt-get install -y sqlite3 \
    && apt-get install -y zip \
    && apt-get install -y python3-opencv nano

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    graphviz \
    libmariadb-dev &&\
    curl -sL https://deb.nodesource.com/setup_18.x | bash - && \
    curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - && \
    echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list && \
    apt-get update && apt-get install -y nodejs yarn

# Redis installation
RUN apt-get install -y lsb-release \
    && curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list \
    && apt-get -y update \
    #&& apt-get install -y redis \
    && rm -rf /var/lib/apt/lists/*

# Set wkdir & Install required packages
COPY ./src/ $APP_ROOT
ENV RAILS_ENV production
RUN yarn install
RUN gem install rails && bundle install
RUN rails db:create db:migrate assets:precompile

# Set user permission
ENV SETUSER_ $SETUSER
#RUN useradd -m $SETUSER_ \
#    && chown -R $SETUSER_:$SETUSER_ $APP_ROOT
RUN chown -R ${USER_ID}:${GROUP_ID} $APP_ROOT
USER $SETUSER_

# Clean cache and Entry with conda mediapipe env
# RUN rails tmp:cache:clear && rails db:reset

ENV CONDA_PKG_ $CONDA_PKG
SHELL ["/bin/bash", "-c"]
RUN source $CONDA_PKG_/bin/activate
EXPOSE 13006
#ENTRYPOINT source $CONDA_PKG_/bin/activate
