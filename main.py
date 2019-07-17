import torch
from sacred import Experiment
from argparse import Namespace
from models.ParaphraseModel import ParaphraseModel
from loader import ParaLoader, pad_collate
from torch.utils.data import DataLoader
import numpy as np
from utils.misc import kld_coef, sentencify




ex = Experiment('1337 Parser')


@ex.config
def cfg():
    """
    General Hyperparameters
    """
    epochs = 10  # Number of epochs to run
    lr = 3e-4  # Learning Rate
    batch_size = 1  # Batch size
    words_dict = 'words_dict.npy'
    train_dataset = 'data/train'
    validation_dataset = 'data/validation'
    test_dataset = 'data/test'
    starting_weights = None
    output_weights = 'weights/bla'
    



def train(epoch, model, dataset, optimizer):
    print('epoch %s train' % epoch)
    model.train()
    total_loss = 0.0
    for i, (inp, output, input_len, output_len, paths) in enumerate(dataset):
        model.zero_grad()
        loss = model.train_model(inp, output, input_len, output_len)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if not i%1000:
            print('processed %s samples\ncurrent loss: %s' % (i, total_loss/(dataset.batch_size*i+1)))
            


    return total_loss / len(dataset)


def validate(epoch, model, dataset):
    print('epoch %s validation' % epoch)
    model.eval()
    total_loss = 0.0
    for i, (inp, output, input_len, output_len, paths) in enumerate(dataset):
        loss = model.train_model(inp, output, input_len, output_len)
        total_loss += loss.item()
        if not i%1000:
            print('processed %s samples\ncurrent loss: %s' % (i, total_loss/(dataset.batch_size*i+1)))
    return total_loss / len(dataset)


@ex.automain
def main(_run):
    args = Namespace(**_run.config)
    
    print(args)
    

    #torch.autograd.set_detect_anomaly(True)
    
    dct = np.load(args.words_dict).item()
    dictionary_size = len(dct)
    
    train_dataset = DataLoader(ParaLoader(args.train_dataset, args.words_dict), batch_size=args.batch_size, collate_fn=pad_collate)
    validation_dataset = DataLoader(ParaLoader(args.validation_dataset, args.words_dict), batch_size=1,
                                    collate_fn=pad_collate)
    test_dataset = DataLoader(ParaLoader(args.test_dataset, args.words_dict), batch_size=args.batch_size, collate_fn=pad_collate)
    m = ParaphraseModel(dictionary_size=dictionary_size).cuda()
    
    if args.starting_weights:
        m.load_state_dict(torch.load(args.starting_weights))
    optimizer = torch.optim.Adam(m.parameters(), lr=args.lr)
    
    
                
    for epoch in range(1, args.epochs + 1):
        train_res = train(epoch, m, train_dataset, optimizer)
        torch.save(m.state_dict(), args.output_weights+'_epoch%s'%epoch)
        val_res = validate(epoch, m, validation_dataset)
        print('train loss: %s\nvalidation loss: %s' % (train_res, val_res))
    
    for (inp, output, input_len, output_len, paths) in test_dataset:
            infres = m.infer(inp, input_len)
            for i in sentencify(infres, dct):
                print('%s\n'%open(paths[0],'r').read(),i)
    
        
        
    
                

