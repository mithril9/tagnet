import sys, os, pdb
from typing import Tuple, List

def main(ccg_folder_path: str) -> None:
    """
    Takes as input the top level CCGbank folder path and creates a new folder ccg_supertag_data which contains
    6 text files, train.words, train.tags, val.words, val.tags, test.words and test.tags.
    train.words and train.tags contain the words and CCGbank lexical tags for all sentences in CCGbank sections 2-22
    (the standard training set); val.words and val.tags contains words and tags from section 00; test.words and
    test.tags correspond to section 23.
    """
    if "ccg_supertag_data" not in os.listdir(os.getcwd()):
        os.mkdir("ccg_supertag_data")
    train_words_file = open(os.path.join("ccg_supertag_data", "train.words"), "w")
    train_tags_file = open(os.path.join("ccg_supertag_data", "train.tags"), "w")
    val_words_file = open(os.path.join("ccg_supertag_data", "val.words"), "w")
    val_tags_file = open(os.path.join("ccg_supertag_data", "val.tags"), "w")
    test_words_file = open(os.path.join("ccg_supertag_data", "test.words"), "w")
    test_tags_file = open(os.path.join("ccg_supertag_data", "test.tags"), "w")
    unique_ccg_tags = set({})
    for folder in os.listdir(ccg_folder_path):
        if folder in ['01', '24']:
            continue
        elif folder == '00':
            word_file, tag_file = val_words_file, val_tags_file
        elif folder == '23':
            word_file, tag_file = test_words_file, test_tags_file
        else:
            word_file, tag_file = train_words_file, train_tags_file
        try:
            int(folder)
        except ValueError:
            continue
        for file_name in os.listdir(os.path.join(ccg_folder_path,folder)):
            if file_name != ".DS_Store":
                for line in open(os.path.join(ccg_folder_path,folder,file_name)):
                    if line[:3] =='ID=':
                        continue
                    words, ccg_supertags = get_words_supertags(line.strip())
                    for supertag in ccg_supertags:
                        unique_ccg_tags.add(supertag)
                    word_file.write(" ".join(words)+"\n")
                    tag_file.write(" ".join(ccg_supertags)+"\n")

    print("num ccg supertags: "+str(len(unique_ccg_tags)))
    train_words_file.close()
    train_tags_file.close()
    val_words_file.close()
    val_tags_file.close()
    test_words_file.close()
    test_tags_file.close()


def get_words_supertags(line: str) -> Tuple[List[str], List[str]]:
    """
    Takes as input a CCGbank file line and returns two lists of strings, one containing each word in the sentence
    and the other containing the CCG lexical categories for the words.
    """
    terminal_open_bracket = False
    terminals = []
    terminal = ""
    for char in line:
        if char == '<':
            terminal_open_bracket = True
            terminal = ""
        elif char == '>':
            terminal_open_bracket = False
            if len(terminal.strip().split()) == 6:
                terminals.append(terminal.strip())
        elif terminal_open_bracket:
            terminal+=char
    words = [terminal.split()[4] for terminal in terminals]
    ccg_supertags = [terminal.split()[1] for terminal in terminals]
    return words, ccg_supertags


if __name__ == "__main__":
    ccg_folder_path = sys.argv[1]
    main(ccg_folder_path)