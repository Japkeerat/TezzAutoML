from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="TezzAutoML",
    version="0.1.5",
    description="Just another AutoML library, but better and faster.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Japkeerat Singh",
    author_email="japkeerat21@gmail.com",
    packages=find_packages(),
    install_requires=required,
    python_requires=">=3.11",
    url="https://github.com/Japkeerat/TezzAutoML",
    license="CC0-1.0",
    keywords="automl machine-learning ml data-science",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        "Programming Language :: Python :: 3.11",
    ]
)
