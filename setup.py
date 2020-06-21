from setuptools import setup
from setuptools import find_packages

setup(name='gcn_mobility_relationhsip',
      version='1.0',
      description='Graph Convolutional Networks on User Mobility Heterogeneous Graphs',
      author='Yongji Wu',
      author_email='wuyongji317@gmail.com',
      license='MIT',
      install_requires=['numpy',
                        'tensorflow',
                        'scipy'
                        ],
      package_data={'heter_gcn': ['README.md']},
      packages=find_packages())