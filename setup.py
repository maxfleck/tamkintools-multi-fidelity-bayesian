from setuptools import setup, find_packages

setup(
    name="tamkintools_multi_fidelity_bayesian",
    version="0.0.1",
    description="A multi fidelity gaussian process and bayesion optimization tool to feed tamkintools wit beautiful data from different levels of theory",
    url="https://github......",
    author="Maximilian Fleck",
    author_email="fleck@itt.uni-stuttgart.de",
    packages=find_packages(),
    install_requires=["emukit","gPy","GPyOpt", "numpy", "matplotlib", "rapidfuzz", "toml", "rdkit"],
)
