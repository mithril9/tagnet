import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tagnet", # Replace with your own username
    version="1.0",
    author="John Torr",
    author_email="john@cantab.net",
    description="A neural network supertagger",
    long_description="A neural network supertagger using  word and character level LSTMs and (optionally) bert word embeddings",
    long_description_content_type="text/markdown",
    url="https://github.com/johnphiliptorr/tagnet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)