IMAGE_REPOSITORY := $(ADAPTDL_DEV_REPO)
REPO_CREDS := $(ADAPTDL_DEV_REPO_CREDS)

NAMESPACE = default
IMAGE_TAG = $(shell cat .devtag)
IMAGE_DIGEST = $(shell docker images --format='{{.Repository}}:{{.Tag}} {{.Digest}}' | \
                       grep '^$(IMAGE_REPOSITORY):$(IMAGE_TAG)' | awk '{ printf $$2 }')
RELEASE_NAME = adaptdl

.devtag:
	@uuidgen > .devtag

.values.yaml:
	@awk '{print "#" $$0}' helm/adaptdl/values.yaml > .values.yaml

check-reg:
ifeq ($(ADAPTDL_DEV_REPO),)
ifeq ($(shell helm ls -q | grep adaptdl-registry),)
			$(error Need a registry, do make deploy-reg or set ADAPTDL_DEV_REPO)
endif
endif

build: .devtag check-reg
	docker build -f docker/Dockerfile . -t $(IMAGE_REPOSITORY):$(IMAGE_TAG)

push: build
	docker push $(IMAGE_REPOSITORY):$(IMAGE_TAG)

deploy: push .values.yaml
	$(info Using $(IMAGE_REPOSITORY) and $(REPO_CREDS))
	helm dep up helm/adaptdl
	helm upgrade $(RELEASE_NAME) helm/adaptdl --install --wait \
		--set image.repository=$(IMAGE_REPOSITORY) \
		--set image.digest=$(IMAGE_DIGEST) \
		--set image.secrets.name=$(REPO_CREDS) \
		--set docker-registry.enabled=true \
		--values .values.yaml

delete: .devtag
	helm delete $(RELEASE_NAME)

pip-install-reqs:
	pip3 install -r adaptdl_cli/requirements.txt

deploy-reg: pip-install-reqs
	$(info Using $(IMAGE_REPOSITORY) and $(REPO_CREDS))
	helm upgrade $(RELEASE_NAME)-registry stable/docker-registry --install --wait \
		--values registry/values.yaml
	python3 adaptdl_cli/adaptdl_cli/registry.py
	kubectl create -f registry/registry-creds.yaml

delete-reg:
	helm delete $(RELEASE_NAME)-registry
	kubectl delete -f registry/registry-creds.yaml

config: .values.yaml
	$(or $(shell git config --get core.editor),editor) .values.yaml

.PHONY: build push deploy delete config deploy-reg delete-reg check-reg pip-install-reqs
