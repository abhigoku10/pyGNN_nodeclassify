from setuptools import setup
from setuptools import find_packages

setup(name='pyGNN',
      version='0.1',
      description='Graph Convolutional Networks in PyTorch',
      author='abhigoku10',
      author_email='abhigoku10@gmail.com',
      url='https://abhigoku10.github.io',
      download_url='https://github.com/abhigoku10/pygcn',
      install_requires=['numpy',
                        'torch',
                        'scipy'
                        ],
      package_data={'pyGNN': ['README.md']},
      packages=find_packages())
