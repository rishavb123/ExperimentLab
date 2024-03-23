./scripts/clean.sh
python -m build
twine upload -r testpypi dist/*