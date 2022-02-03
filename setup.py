"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
from glob import glob
from os.path import basename, splitext

import setuptools


requirements = [
    "numpy==1.18.1",
    "opencv-python==4.2.0.32",
]

setuptools.setup(
    name="nst",
    version="0.0.1",
    install_requires=requirements,
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires="~=3.7",
)
