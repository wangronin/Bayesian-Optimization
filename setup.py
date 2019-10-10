from setuptools import setup, find_packages
setup(
    name = 'bayesopt',
    version = '0.1',
    #packages = find_packages(),
    packages = ["BayesOpt", "BayesOpt.optimizer"],
    #Metadata
    author = 'wangronin',
    description = 'Experimental Bayesian Optimization library',
    license = '',
    url = 'https://github.com/wangronin/Bayesian-Optimization'
)
