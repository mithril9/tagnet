#Tagnet
=========

This repository contains code for a recently reimplemented version of the neural network supertagger described in the ACL 2019 paper "Wide-Coverage Neural A* Parsing for Minimalist Grammars", authored by John Torr, Miloš Stanojević, Mark Steedman and Shay Cohen.  The supertagger can be used for any token classification task, including part of speech tagging, named entity tagging, or supertagging.  Please direct any questions about the code in this repo to John Torr (john.torr@cantab.net).

##Installation of the supertagger
---------------

###Basic requirements

The supertagger requires python 3 (it was built and tested using python 3.6.5).

You can install python 3 using the following command.

```
brew install python
```

###Additional dependencies

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



