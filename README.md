# Tagnet

This repository contains code for a recent Pytorch reimplementation of the neural network supertagger described in the ACL 2019 paper [Wide-Coverage Neural A* Parsing for Minimalist Grammars](https://www.aclweb.org/anthology/P19-1238.pdf).  The supertagger can be used for any token classification task, including part of speech tagging, named entity tagging, or supertagging.  The supertagger uses both character and word level bi-LSTMs and optionally uses bert word embeddings.  Please direct any questions about the code in this repo to John Torr (john.torr@cantab.net).

## Installation of the supertagger
---------------

### Basic requirements

The supertagger requires python 3 (it was built and tested using python 3.6.5).

You can install python 3 by opening up a terminal window and executing the following command.

```
brew install python
```

### Additional dependencies

The supertagger has several other dependencies.  You can easily install them all by  executing the following from a terminal from within the tagnet/ folder.

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

In case there are any future compatibility issues with these dependencies, you can try installing the exact versions of the above that the code in this repo was tested with by executing one or more of the following (note that the install.sh script will just install the most recent version of each dependency):

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

For training the supertagger you will need to prepare separate training and validation data sets, and you may also want a separate test set to get a completely unbiased evaluation of your model.  Each dataset should be split into two files, one of which contains the sentences to be tagged and the other contains the tags themselves.  Examples of these files are included in the folder tagnet/data/dummy where you will note that there are the following six files:

train.tags
train.words
test.tags
test.words
val.tags
val.words

Your files should be named exactly the same as above.

The .words files contain one (tokenized) sentence per line, each word separated by a single space.  The .tag files contain a tag sequence on each line, again with each tag being separated by a single space.  The number of tags on line i of file x.tags should be exactly equal to the number of words on line i of file x.words.  Note that the example files all contain exactly the same sentences as each other, but of course in reality you will have different sentences in each of the files.

Note that the supertagger was tested on CCGbank supertagging and a script is included for converting the original CCGbank files into the required .train, .test and .val files.  To run it, first download CCGbank from the [linguistic data consortium website](https://catalog.ldc.upenn.edu/LDC2005T13) and move the ccg/ folder into tagnet/ and then execute the following:

```
python getCCGsupertags.py ccg/
```

Note that if your computer is set up to use Python 2 by default, you will need use 'python3' instead of just 'python', so that the above line would then be:

```
python3 getCCGsupertags.py ccg/
```

This will create a folder named ccg_supertag_data/ which contains all 6 of the required files using the standard test/validation/test split from the literature on CCG parsing and supertagging.

### Training

To train a model from scratch, and assuming your training data is inside tagnet/data/dummy, you would first cd into tagnet/supertagger and then execute the following command:

```
python train.py --data-path ../data/dummy
```

You can edit the hyper parameters of the model by editing the file tagnet/supertagger/config.py.  You can either train a model using the final hidden layer of a [bert model](https://arxiv.org/pdf/1810.04805.pdf) as word embeddings (the bert model will be frozen, i.e. it's parameters will not be updated as the model learns) or you can start with randomly initialized word embeddings and allow the system to try to learn the embeddings itself.  Bert works best but training will be slower and memory requirements greater.

To turn bert on, in config.py, either use_bert_uncased or use_bert_cased (but not both) must be set to True.  In addition, you can choose whether to use the bert base or bert large model by setting use_bert_large to True or False.  Setting this to True will not result in bert being used unless either use_bert_uncased or use_bert_cased is also set to True.

As your model is training it will be saved inside tagnet/models/ and the saved model's parameters will be updated whenever a lower loss is achieved on the valuation dataset.  To resume training of a peviously saved model, you can execute:

```
python train.py --data-path ../data/dummy --model-path ../models/<model-name>
```

For example, if the model name is 2020_06_20_17_03_05.pt, you would execute:

```
python train.py --data-path ../data/dummy --model-path ../models/2020_06_20_17_03_05.pt
```

This will pick up from whichever epoch on the previous training round had the lowest val loss.  It will not overwrite the original model, however; if a better loss on the eval set is achieved, the resulting model will be saved as a separate file.

### Testing

To test a saved model on the test set, execute the following:

```
python test.py --data-path ../data/dummy --model-path ../models/2020_06_20_17_03_05.pt
```

### Tagging new data

You can also use the model to tag data for which you do not have any tags.  Let's say you have some sentences in data/untagged/sentences.words, and you want to save the predictions in data/unseen/predictions.  Then you would execute:

```
python tag.py --data-path ../data/untagged/sentences.words --dest-path ../data/untagged/predictions --model-path ../models/2020_06_20_17_03_05.pt
```

Notice that for tag.py, unlike for train.py and test.py, the argument to --data-path includes the actual file name itself, rather than just the path to the folder containing the file.  The reason for this is that you can instruct tag.py to tag sentences contained inside a file with any name, whereas as noted above, for train.py and test.py the file names are fixed in advance.


