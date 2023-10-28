.PHONY: all coverage clean docs_compile docs_build docs_clean docs_linkcheck format lint test tests test_watch help

all: help

coverage:
	poetry run pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered

clean: docs_clean

docs_compile:
	poetry run nbdoc_build --srcdir $(srcdir)

docs_build:
	cd docs && poetry run make html

docs_clean:
	cd docs && poetry run make clean

docs_linkcheck:
	poetry run linkchecker docs/_build/html/index.html

format:
	poetry run black .
	poetry run ruff --select I --fix .

PYTHON_FILES=.
lint: PYTHON_FILES=.
lint_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep | -E '\.py$$')

lint lint_diff:
	poetry run mypy $(PYTHON_FILES)
	poetry run black $(PYTHON_FILES) --check
	poetry run ruff .

TEST_FILE ?= tests/unit_tests/

test:
	poetry run pytest --disable-socket --allow-unix-socket $(TEST_FILE)

tests:
	poetry run pytest --disable-socket --allow-unix-socket $(TEST_FILE)

test_watch:
	poetry run ptw --now . -- tests/unit_tests
