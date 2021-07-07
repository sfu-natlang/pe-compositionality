import os
import re
from io import open
import torch

from PIL import Image, ImageOps
from torchvision import transforms

VERBOSE = False

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counts   = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.counts[word] = 0
        self.counts[word] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class ImageDictionary(Dictionary):
    def __init__(self, size, base_path):
        super().__init__()
        self.memo = {}
        self.size = size
        self.base_path = base_path

    def get_image(self, idx):
        word = self.idx2word[idx]

        if word not in self.memo:
            vec = torch.zeros( (1, self.size, self.size) )
            path = None

            filename = word + ".png"
            path = os.path.join(self.base_path, filename)

            # fall back on other variants if image is missing
            # construct a list of candidates in rough order
            # of how likely they are, and use the first one that
            # has an image file:
            if not os.path.isfile(path):
                candidates = []

                if word[-1].isdigit():
                   # from MXXX-A1, fall back on MXXX-A
                   candidates.append(word[:-1])

                if "-" in word:
                   # from MXXX-A, fall back on MXXX
                   candidates.append( re.sub( "-[A-Z0-9]*", "", word) )

                if "@" in word:
                   # from N(NXX@B), fall back on N(NXX)
                   candidates.append( re.sub( "\@[A-Z0-9]*", "", word) )

                # scan directory for filenames which might
                # represent suitable signs:
                dirlist = os.listdir(self.base_path)
                for w in dirlist:
                    w = w.replace(".png", "")

                    if w.startswith( re.sub( "-[A-Z0-9]*", "", word) ) and "+" not in w:
                       # from MXXX, fall back on MXXX-A (but not MXXX+MYYY)
                       candidates.append( w )

                    if "+" in word and re.sub( "-[A-Z0-9]*", "", word ) == re.sub( "-[A-Z0-9]*", "", w ):
                       # from MXXX-A+MYYY-B, fall back on MXXX+MYYY
                       candidates.append( w )

                    if "(N" in word and "M" not in word and re.sub( r'.*\((N[^)]*)\).*', "\\1", word ) == re.sub( r'.*\((N[^)]*)\).*', "\\1", w ):
                       # from N(NXX), fall back on 1(NXX)
                       candidates.append( w )

                for candidate in candidates:
                    path = os.path.join(self.base_path, candidate + ".png")
                    if os.path.isfile(path):
                        break

            if os.path.isfile(path):
                vec = Image.open(path).resize( (self.size, self.size) )
                vec = ImageOps.invert(vec)
                vec = transforms.ToTensor()(vec)
            elif VERBOSE:
                print("WARN: no image for", word)

            self.memo[word] = vec
        return self.memo[word]


class Corpus(object):
    def __init__(self, path, image_path, image_size=64):
        self.dictionary = ImageDictionary(image_size, image_path)
        self.train = self.tokenize( path.replace("txt", "train" ))
        self.valid = self.tokenize( path.replace("txt", "valid" ))
        self.test = self.tokenize( path.replace("txt", "test" ))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
