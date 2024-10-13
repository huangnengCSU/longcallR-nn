from setuptools import setup, find_packages

version = {}
with open("longcallR_nn/_version.py") as version_file:
    exec(version_file.read(), version)

setup(
    name='longcallR_nn',
    version=version['__version__'],
    author="Neng Huang",
    license="Licensed under the MIT License",
    author_email="neng@ds.dfci.harvard.edu",
    long_description="longcallR: a deep learning based variant caller for long-reads RNA-seq data",
    url="https://github.com/huangnengCSU/longcallR-nn",
    download_url='https://github.com/huangnengCSU/longcallR-nn/archive/{}.tar.gz'.format(version['__version__']),
    entry_points={
        'console_scripts': [
            'longcallR_nn=longcallR_nn.longcallR_nn:main',
        ],
    },
    platforms="Unix like",
    zip_safe=False,
    packages=find_packages(),  # Make sure this includes all packages
)
