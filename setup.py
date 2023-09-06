from setuptools import setup, find_packages

setup(
    name='SOM_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'viffIO>=0.0.1',
        'numpy>=1.22.4',
        'strax>=1.5.2',
        'matplotlib>=3.7.1'
        # list of dependencies, e.g., 'numpy>=1.18',
    ],
)