## Local Development Guide

### Installation
- Install [docker](https://docs.docker.com/engine/install/)
- Install Makefile

    [For windows](https://stackoverflow.com/questions/2532234/how-to-run-a-makefile-in-windows)
    - Install the chocolatey package manager for Windows
    - Run choco install make

    For linux:
    ``` bash 
    sudo apt install build-essential
    make -version
    ``` 

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

###  For running without any local changes
``` bash 
make run
``` 
Open http://localhost:7777 on browser. Port can be changed in Makefile if required.
###  For running with the local changes
``` bash 
make run-local-dir
``` 
Open http://localhost:7777 on browser. Port can be changed in Makefile if required.