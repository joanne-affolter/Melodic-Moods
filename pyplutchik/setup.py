from setuptools import setup, find_packages

setup(
    name='pyplutchik',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'Shapely==1.7.1',
        'numpy==1.17.4',
        'matplotlib==3.3.2',
    ],
    # Include other arguments as needed
)
