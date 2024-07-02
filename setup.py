from setuptools import setup, find_packages

setup(
    name='smplx',
    version='0.0.1',
    description='smplx tools', 
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm', 
        # 'pickle', 
    ],
)