from pathlib import Path
import numpy as np
import os

# TODO: Add tokenization, stemming lemmatization (spacy or NLTK)

words = {}

words['<SOS>'] = 0
words['<EOS>'] = 1
words['<UNK>'] = 2

counter = 3
for f_name in Path('data').glob('**/*'):
    if os.path.isdir(f_name):
        continue

    with open(f_name, 'r') as f:

        try:
            i, o = f.read().split(r'|||')

            for word in i.split() + o.split():
                if word in words:
                    continue
                else:
                    words[word] = counter
                    counter += 1

        except:
            f.close()
            print(f_name)
            os.rename(f_name, 'badencoding_data/'+f_name.name)

#words['<SOS>'] = counter
#words['<EOS>'] = counter + 1
#print(len(words))
np.save('words_dict.npy', words)
