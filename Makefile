jn:
	poetry run jupyter lab

test:
	@make test-pytest

test-pytest:
	poetry run pytest -s --pdb tests/
