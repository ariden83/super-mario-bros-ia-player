VENV := .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

.PHONY: run install help

run: $(VENV)/.deps_installed ## Vérifie les dépendances et lance Mario
	$(PYTHON) mario_fluid_llm.py

$(VENV)/.deps_installed: requirements.txt
	@if [ ! -d "$(VENV)" ]; then \
		echo "Création de l'environnement virtuel..."; \
		python3 -m venv $(VENV); \
	fi
	@echo "Installation des dépendances..."
	$(PIP) install -r requirements.txt
	@touch $(VENV)/.deps_installed
	@echo "Dépendances prêtes."

install: ## Réinstalle toutes les dépendances
	@rm -f $(VENV)/.deps_installed
	@$(MAKE) $(VENV)/.deps_installed

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
