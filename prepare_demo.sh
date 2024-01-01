#!/bin/bash

docker compose down
docker image rm skinx-paul-sentiment:0.0.1
docker build dockerfile -t skinx-paul-sentiment:0.0.1

