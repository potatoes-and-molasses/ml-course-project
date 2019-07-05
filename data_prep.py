import pandas as pd
import spacy
import argparse
import os
import shutil

sp = spacy.load('en_core_web_sm') 


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', help='data csv file path')
parser.add_argument('-r', '--root', help='root path for data dir')
parser.add_argument('-c', '--clear', action='store_true', help='remove existing data dirs')
parser.add_argument('-tr', '--train', help='train sample size')
parser.add_argument('-v', '--validation', help='validation sample size')
parser.add_argument('-te', '--test', help='test sample size')

args =  parser.parse_args()

if args.root:
    if args.clear:
        shutil.rmtree(args.root)
    dirs = [args.root+'/%s'%i for i in ['','train/','test/','validation/']]
    for i in dirs:
        if not os.path.isdir(i):
            os.mkdir(i)

train_size = int(args.train)
validation_size = int(args.validation)
test_size = int(args.test)
dataset_path = args.data

dataset = pd.read_csv(dataset_path)


positive = dataset[dataset.is_duplicate==1]


stuff = positive.sample(train_size+validation_size+test_size)

train, validation, test = stuff[:train_size], stuff[train_size:train_size+validation_size], stuff[-test_size:]

def prep_sentence(snt):

    snt = sp(snt)
    split = ' '.join(str(i) for i in snt)#temp.. for now not removing anything, just splitting into parts according to language model, is it worth removing stuff? maybe punctuation and such are important for the model?
    #split = ' '.join(str(i) for i in snt if not i.is_punct)#removing punctuation marks option?
    
    
##    ents = {str(i):str(i).replace(' ','_') for i in snt.ents}#joining entities into 1 token(i.e. "General Electic" -> "General_Electric")
##    for i in ents:
##        split = split.replace(i, ents[i])

    return split
        
        

def pd_to_samples_dir(data, path):
    for row in data.iterrows():
        idx, question, rephrase = row[1].id, row[1].question1, row[1].question2
        question, rephrase = prep_sentence(question), prep_sentence(rephrase)
        with open(path+'/%s'%idx,'w') as f:
            f.write(question+'|||'+rephrase)
            
pd_to_samples_dir(train, '%s/%s'%(args.root, 'train'))
pd_to_samples_dir(validation, '%s/%s'%(args.root, 'validation'))
pd_to_samples_dir(test, '%s/%s'%(args.root, 'test'))

        



