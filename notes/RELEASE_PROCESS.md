bumpverion patch
python -m build

pip install auditwheel
sudo apt-get install patchelf
auditwheel repair dist/rfnetwork-0.1.0-cp312-cp312-linux_x86_64.whl -w dist/
twine upload dist/*

To run docs
sphinx-build -b html docs docs/build