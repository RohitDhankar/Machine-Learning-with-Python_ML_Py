# SOURCE -- https://realpython.com/natural-language-processing-spacy-python/#installation-of-spacy

import spacy

nlp = spacy.load("en_core_web_sm")
print(type(nlp)) ##<class 'spacy.lang.en.English'>

"""
To start processing your input, you construct a Doc object.
 A Doc object is a sequence of Token objects representing a lexical token.
  Each Token object has information about a 
  particular piece—typically one word—of text.
"""

introduction_doc = nlp("This tutorial is about Natural Language Processing in spaCy.")
print(type(introduction_doc)) # <class 'spacy.tokens.doc.Doc'>

ls_test = [token.text for token in introduction_doc]
print(ls_test)
#token == TOKEN Object 
#.text attribute to get the text contained within that token.
# ['This', 'tutorial', 'is', 'about', 'Natural', 'Language', 'Processing', 'in', 'spaCy', '.']
# Token object, you called the .text attribute to get the text contained within that token.

import pathlib
#file_name = "/home/dhankar/temp/08_23/spacy_1/ner_1.log"
file_name = "/home/dhankar/temp/08_23/spacy_1/ner_2.log"

introduction_doc = nlp(pathlib.Path(file_name).read_text(encoding="utf-8"))
#print([token.text for token in introduction_doc]) # OK 
#
sentences = list(introduction_doc.sents)
print(len(sentences))
for sentence in sentences:
    print(f"{sentence[:5]}   ----> FIRST FIVE TOKENS ONLY")

# Custom - -+---> set_custom_boundaries
from spacy.language import Language
@Language.component("set_custom_boundaries")
def set_custom_boundaries(introduction_doc):
    """
    """
    for token in introduction_doc[:-1]:
        if token.text == "-+--->":
            introduction_doc[token.i + 1].is_sent_start = True
    return introduction_doc

custom_nlp = spacy.load("en_core_web_sm")

