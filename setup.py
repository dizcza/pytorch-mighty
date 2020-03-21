# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import setup

with open(Path(__file__).with_name("mighty") / "VERSION") as version_file:
    version = version_file.read().strip()

with open("README.md") as f:
    long_description = f.read()
with open('requirements.txt') as f:
    install_requires = f.read()


setup(
    name="pytorch-mighty",
    version=version,
    packages=['mighty'],
    include_package_data=True,
    install_requires=install_requires,
    author="Danylo Ulianych",
    author_email="d.ulianych@gmail.com",
    description="The Mighty Monitor Trainer for your pytorch models.",
    long_description=long_description,
    license="BSD-3",
    url='https://github.com/dizcza/pytorch-mighty',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
    ]
)
