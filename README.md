Tagnet
=========

This repository contains code for a recently reimplemented version of the neural network supertagger described in the ACL 2019 paper "Wide-Coverage Neural A* Parsing for Minimalist Grammars", authored by John Torr, Miloš Stanojević, Mark Steedman and Shay Cohen.  The supertagger can be used for any token classification task, including part of speech tagging, named entity tagging, or supertagging.  Please direct any questions about the code in this repo to John Torr (john.torr@cantab.net).

Installation of the supertagger
---------------

The supertagger has several 3rd party dependencies.  You can easily install them all by  executing the following from a terminal from within the tagnet/ folder.

```
./install.sh
```

Alternatively, if you prefer to install the dependencies separately, execute the following from a terminal window:

```
pip install torch
pip install numpy
pip install sklearn
pip install torchtext
pip install pandas
pip install matplotlib
pip install nltk
pip install transformers
```

