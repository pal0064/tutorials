## Local Development Guide

### Installation
- Install [docker](https://docs.docker.com/engine/install/)
- Install Makefile

    [For Windows](https://stackoverflow.com/questions/2532234/how-to-run-a-makefile-in-windows)
    - Install the chocolatey package manager for Windows
    - Run choco install make

    For linux:
    ``` bash 
    sudo apt install build-essential
    make -version
    ``` 

### For directly executing without the `Makefile`
``` bash
IMAGE_NAME=palankit0064/nlp-tutorials
docker run -it -p 7777:8888 ${IMAGE_NAME}:latest
```
Open http://localhost:7777 on your browser

###  For running without any local changes
``` bash 
make run
``` 
Open http://localhost:7777 on your browser. Port can be changed in Makefile if required.

###  For running with the local changes
``` bash 
make run-local-dir
``` 
Open http://localhost:7777 on your browser. Port can be changed in Makefile if required.

###  For building
Change IMAGE_NAME in Makefile

``` bash 
make build
``` 
###  For publishing
Change IMAGE_NAME in Makefile. Also, you have to [login](https://docs.docker.com/engine/reference/commandline/login/) using docker account for publishing.

``` bash 
make push
``` 
