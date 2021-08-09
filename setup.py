from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bayes-optim",
    version="0.2.5",
    author="Hao Wang",
    author_email="wangronin@gmail.com",
    packages=find_packages(),
    description="A Bayesian Optimization Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wangronin/Bayesian-Optimization",
    package_dir={"bayes_optim": "bayes_optim"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    download_url="https://github.com/wangronin/Bayesian-Optimization/archive/v0.1.3.tar.gz",
    python_requires=">=3.6",
    install_requires=[
        "dill>=0.3.2",
        "joblib>=0.16.0",
        "numpy>=1.19.0",
        "pyDOE>=0.3.8",
        "scikit-learn>=0.23.0",
        "scipy>=1.5.0",
        "sklearn==0.0",
        "tabulate>=0.8.7",
        "threadpoolctl>=2.1.0",
        "torch>=1.7.1",
    ],
)
