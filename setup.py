from setuptools import setup, find_packages
description = 'Official Bookclub Client'

with open("./requirements.txt", "r", encoding="utf-8") as f:
    install_requires = f.read()

setup(
    version="0.0.1",
    install_requires=install_requires,
    description=description
)