from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="TezzAutoML",
    version="0.1.0",
    description="An AutoML library built on top of Optuna.",
    author="Japkeerat Singh",
    author_email="japkeerat21@gmail.com",
    packages=find_packages(),
    install_requires=required,
)
