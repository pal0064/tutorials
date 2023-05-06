IMAGE_NAME = palankit0064/nlp-tutorials
IMAGE_VERSION = 0.1
ARCH = linux/386,linux/arm64,linux/amd64

.PHONY: build run run-local-dir push convert-notebook

build:
	-docker buildx rm -f mybuilder
	docker buildx create --name mybuilder --driver docker-container --bootstrap
	docker buildx use mybuilder
	docker buildx build --platform ${ARCH} -t ${IMAGE_NAME}:${IMAGE_VERSION} -t ${IMAGE_NAME}:latest .

run:
	docker run -it -p 7777:8888 ${IMAGE_NAME}:latest

run-local-dir:
	docker run -it -p 7777:8888 -v "$PWD:/app/" ${IMAGE_NAME}:latest

push:
	docker buildx build --platform ${ARCH} -t ${IMAGE_NAME}:${IMAGE_VERSION} -t ${IMAGE_NAME}:latest --push .

convert-notebook:
	jupyter nbconvert --to markdown stanza/Stanza.ipynb