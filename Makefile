.ONESHELL:

.PHONY: clean install tests run all

clean:
	find . -type f -name '*.pyc' -prune -exec rm -rf {} \;
	find . -type f -name '*.log' -prune -exec rm -rf {} \;
	find . -type f -name '*.db' -prune -exec rm -rf {} \;
	find . -type d -name "__pycache__" -prune -exec rm -rf {} \;
	find . -type d -name "migrations" -prune -exec rm -rf {} \;

install:
	python -m pip install --upgrade pip;
	pip install -r requirements.txt;

migration:
	python manage.py db init;
	python manage.py db migrate --message 'initial database migration';
	python manage.py db upgrade;

tests:
	python manage.py test

run:
	python manage.py run

all: clean install migration tests run
