# Tagnet

This repository contains code for a recently reimplemented version of the neural network supertagger described in the ACL 2019 paper "Wide-Coverage Neural A* Parsing for Minimalist Grammars", authored by John Torr, Miloš Stanojević, Mark Steedman and Shay Cohen.  The supertagger can be used for any token classification task, including part of speech tagging, named entity tagging, or supertagging.  Please direct any questions about the code in this repo to John Torr (john.torr@cantab.net).

## Installation of the supertagger
---------------

### Basic requirements

The supertagger requires python 3 (it was built and tested using python 3.6.5).

You can install python 3 using the following command.

```
brew install python
```

### Additional dependencies

The supertagger has several other 3rd party dependencies.  You can easily install them all by  executing the following from a terminal from within the tagnet/ folder.

```
./install.sh
```

Alternatively, if you prefer to install the dependencies separately, execute the following from a terminal window:

```
pip install matplotlib
pip install nltk
pip install numpy
pip install pandas
pip install sklearn
pip install torch
pip install torchtext
pip install transformers
```

In case there are any future compatibility issues with these dependencies, you can try install the exact versions of the above I am currently using by executing one or more of the following (note that the install.sh script will just install the most recent version of each dependency) :

```
pip install matplotlib==3.1.1
pip install nltk==3.5
pip install numpy==1.18.5
pip install pandas==1.0.0
pip install sklearn==0.0
pip install torch==1.5.1
pip install torchtext==0.6.0
pip install transformers==2.11.0
```

## Usage

### Data Preparation

For training the supertagger you will need to prepare separate training and validation data sets, and you may also want a separate test set to get a completely unbiased evaluation of your model.  Each dataset should be split into two files, one of which contains the sentences to Examples of these files are included in the folder tagnet/data/dummy where you will note that there are the following six files:

train.tags
train.words
test.tags
test.words
val.tags
val.words

The .words files contain one (tokenized) sentence per line, each word separated by a single space.  The .tag files contain a tag sequence on each line, again with each tag being separated by a single space.  The number of tags on line i of x.tags should be exactly equal to the number of words on line i of x.words.  The above example files all contain exactly the same sentences, but of course in reality you would have different sentences in each of the files.

Note that the supertagger was tested on CCGbank and a script is included for converting the CCGbank files into the required .train, .test and .val files.  To run it, first download CCGbank from the [linguistic data consortium website](https://catalog.ldc.upenn.edu/LDC2005T13) and move the ccg/ folder into tagnet/ and execute the following:

```
python getCCGsupertags.py ccg/
```

This will create a folder named ccg_supertag_data/ which contains all 6 of the required files using the standard test/validation/test split from the literature on CCG parsing and supertagging.

### Training



