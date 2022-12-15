# Create the base OS image
FROM ruby:3.1.2 AS base

#LABEL version='1.0'
#LABEL maintainer='<yuy@cmdm.csie.ntu.edu.tw>'

# Update the OS ubuntu image
RUN apt-get -y update \
    && apt-get -y upgrade \
    && rm -rf /var/lib/apt/lists/*

# Install required packages
RUN gem install rails
# apt install sqlite3


# Create another image layer on top of base to install requirements
#FROM base AS docker.io/continuumio/miniconda3:4.12.0
