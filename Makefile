# Makefile

GREEN=\033[0;32m
RED=\033[0;31m
YELLOW=\033[0;33m
RESET=\033[0m


SHELL := /bin/bash
VENV := .venv
PYTHON_VERSION := $(shell grep -m 1 'requires-python' pyproject.toml | awk -F'>=' '{print $$2}' | tr -d '[:blank:]' | tr -d '"')

.PHONY: all check_uv

# check if uv is installed, if not -> install
check_uv:
	@if ! command -v uv > /dev/null ; then \
		echo -e "$(YELLOW) [+] uv not installed. Installing now... $(RESET)"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo -e "$(GREEN) [~] uv installed. Skipping... $(RESET)"; \
	fi

install_python: check_uv
	@if ! command uv python list | grep $(PYTHON_VERSION) | grep -m 1 -q "/"; then \
		echo -e "$(YELLOW) [+] Installing required python version: $(PYTHON_VERSION) using uv... $(RESET)"; \
		uv python install $(PYTHON_VERSION); \
	else \
		echo -e "$(GREEN) [~] python $(PYTHON_VERSION) installed. Skipping... $(RESET)"; \
	fi

venv: install_python
	@if [[ ! -d $(VENV) ]]; then \
		echo -e "$(YELLOW) [+] No venv found. Installing venv using uv... $(RESET)"; \
		uv venv -p $(PYTHON_VERSION); \
	else \
		echo -e "$(GREEN) [~] venv present. Skipping... $(RESET)"; \
	fi

install_deps: venv
	@echo -e "$(YELLOW) [+] Updating and installing deps... $(RESET)"; \
	source $(VENV)/bin/activate && \
	uv pip install --upgrade pip && \
	uv pip install -Ur dev-requirements.txt; \

clean_venv:
	@if [[ -d $(VENV) ]]; then \
		echo -e "$(RED) [!] Deleting venv... $(RESET)"; \
		rm -rf $(VENV); \
	else \
		echo -e "$(GREEN) [~] No $(VENV) installed. Skipping... $(RESET)"; \
	fi
clean_pycache:
	@echo -e "$(RED) [!] Deleting all __pycache__ directories... $(RESET)"; \
	find . -type d -name "__pycache__" -exec rm -rf {}  +; \

clean_python:
	@echo -e "$(RED) [!] Deleting python $(PYTHON_VERSION)... $(RESET)"; \
	uv python uninstall $(PYTHON_VERSION); \

clean_uv:
	@echo -e "$(RED) [!] Deleting uv... $(RESET)"; \
	uv cache clean; \
	rm -rf "$(shell uv python dir)"; \
	rm ~/.local/bin/uv ~/.local/bin/uvx; \

clean: clean_venv clean_pycache clean_python clean_uv

deps_dev: check_uv install_python
	@echo -e "$(YELLOW) [+] Compiling list of dependencies DEV... $(RESET)"; \
	uvx --from pip-tools pip-compile -r -o dev-requirements.txt --extra=dev pyproject.toml; \

deps_prod: check_uv install_python
	@echo -e "$(YELLOW) [+] Compiling list of dependencies PROD... $(RESET)"; \
	uvx --from pip-tools pip-compile -r -o requirements.txt pyproject.toml; \

deps: deps_dev deps_prod
