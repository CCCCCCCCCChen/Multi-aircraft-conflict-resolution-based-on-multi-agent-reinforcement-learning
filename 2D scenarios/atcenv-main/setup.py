import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='Conflict resolution environment',
    version='0.1',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Ramon Dalmau-Codina',
    author_email='ramon.dalmau-codina@eurocontrol.int',
    url='https://github.com/ramondalmau/atcenv.git',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
