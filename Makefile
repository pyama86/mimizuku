clean:
	rm -rf mimizuku.egg-info/* dist/*
build: clean
	python setup.py sdist bdist_wheel

setup:
	pip install twine setuptools wheel

upload:
	twine upload --repository pypi dist/*
