# Set ADAPTDL_DEV_REPO to use an external docker registry.
# Set ADAPTDL_DEV_REPO_CREDS to the name of registry secret.
RELEASE_NAME = adaptdl
LOCAL_PORT = 59283
REMOTE_PORT = 32000
LOCAL_REPO = $(or $(ADAPTDL_DEV_REPO),localhost:$(LOCAL_PORT)/adaptdl-sched)
REMOTE_REPO = $(or $(ADAPTDL_DEV_REPO),localhost:$(REMOTE_PORT)/adaptdl-sched)
IMAGE_TAG = latest
IMAGE_DIGEST = $(shell docker images --format='{{.Repository}}:{{.Tag}} {{.Digest}}' | \
                       grep '^$(LOCAL_REPO):$(IMAGE_TAG) ' | awk '{ printf $$2 }')
NAMESPACE = $(or $(shell kubectl config view --minify -o 'jsonpath={..namespace}'),default)

.values.yaml:
	@awk '{print "#" $$0}' helm/adaptdl-sched/values.yaml > .values.yaml

registry:
	helm status adaptdl-registry || \
	helm install adaptdl-registry stable/docker-registry \
		--set fullnameOverride=adaptdl-registry \
		--set service.type=NodePort \
		--set service.nodePort=$(REMOTE_PORT)

build:
	docker build -f sched/Dockerfile . -t $(LOCAL_REPO):$(IMAGE_TAG)

check-requirements:
	@python3 cli/check_requirements.py

push: check-requirements registry build
	python3 cli/adaptdl_cli/proxy.py -p $(LOCAL_PORT) $(NAMESPACE) \
		adaptdl-registry:registry docker push $(LOCAL_REPO):$(IMAGE_TAG)

deploy: push .values.yaml
	helm dep up helm/adaptdl-sched
	helm upgrade $(RELEASE_NAME) helm/adaptdl-sched --install --wait \
        $(and $(ADAPTDL_DEV_REPO_CREDS),--set 'image.pullSecrets[0].name=$(ADAPTDL_DEV_REPO_CREDS)') \
		--set image.repository=$(REMOTE_REPO) \
		--set image.digest=$(IMAGE_DIGEST) \
		--values .values.yaml

delete:
	helm delete $(RELEASE_NAME) || \
	helm delete adaptdl-registry

config: .values.yaml
	$(or $(shell git config --get core.editor),editor) .values.yaml

.PHONY: registry build push deploy delete config
