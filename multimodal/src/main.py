# coding: utf-8
#
# This code is adapted from the PyTorch language modeling examples
# at https://github.com/pytorch/examples/tree/master/word_language_model

import argparse
import time
import math
import os

import data
import model

import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description='Multimodal language modeling architecture for decipherment')

parser.add_argument('--verbose', action='store_true',
                    help='if true, print all warnings')

parser.add_argument('--train', action='store_true',
                    help='if true, train the model')
parser.add_argument('--export', action='store_true',
                    help='if true, export embeddings to file')

parser.add_argument('--data', type=str, default='./data/clean.txt',
                    help='location of the data corpus')
parser.add_argument('--images', type=str, default='./data/images/pe_64',
                    help='location of the input images')

parser.add_argument('--bidirectional', action='store_true',
                    help='if true, use a bidirectional LSTM')
parser.add_argument('--visual', action='store_true',
                    help='whether to use image inputs')
parser.add_argument('--textual', action='store_true',
                    help='whether to use text inputs')

parser.add_argument('--imgdim', type=int, default=32,
                    help='size of image inputs (pixels per side)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')

parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')

parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda and args.verbose:
        print("WARN: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data, args.images, args.imgdim)

def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

train_data = batchify(corpus.train, args.batch_size)
val_data   = batchify(corpus.valid, args.eval_batch_size)
test_data  = batchify(corpus.test,  args.eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(
          ntokens, 
          args.emsize, 
          args.nhid, 
          args.nlayers, 
          args.textual, 
          args.visual, 
          args.imgdim, 
          args.dropout,
          args.bidirectional
        ).to(device)

def NLLLossIgnoringUNK(unk_idx):
    """
    We do not want the model to use any information from X or ... tokens, as these
    represent breaks in the input text rather than linguistic information and cause
    the model to learn worse embeddings for the other signs. This loss function ensures
    losses are not propagated from these tokens.
    """
    def loss(input, target):
        indices = [i for i in range(target.shape[0]) if target[i].item() != unk_idx]
        return nn.NLLLoss()(input[indices], target[indices])
    return loss
criterion = NLLLossIgnoringUNK(corpus.dictionary.word2idx["UNK"])

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    """

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    # Get longest sequence possible from the source data:
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    # Placeholder for image data:
    image_data = torch.zeros( 
        (seq_len,                 # sequence length
         source[0].shape[0],      # batch size
         1,                       # channels (only one: B/W)
         args.imgdim, args.imgdim # image size
        ), dtype=torch.float32 ).to(device)
    # Fill placeholder data with actual sign images:
    for seq_pos, batch in enumerate(data):
        image_data[seq_pos] = torch.stack([
            corpus.dictionary.get_image(word.item()) 
            for word in batch
        ])
    target = source[i+1:i+1+seq_len].view(-1)
    return data, image_data, target

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, image_data, targets = get_batch(data_source, i)
            output, hidden = model( (data, image_data), hidden)

            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, image_data, targets = get_batch(train_data, i)

        # Each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.

        model.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model( (data, image_data), hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

if args.train:
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

###############################################################################
# Export embeddings
###############################################################################


def save_embeddings(path):
    with open(path+".vec", "w+") as fp:
        for word, idx in corpus.dictionary.word2idx.items():
            if args.textual:
                emb_t = model.text_encoder(torch.LongTensor([idx]).to(device)).cpu().detach()#.numpy().reshape((-1,))
            if args.visual:
                image = torch.unsqueeze(torch.unsqueeze(corpus.dictionary.get_image(idx),0),0)
                emb_v = model.image_encoder(image.to(device)).cpu().detach()[0]#.numpy().reshape((-1,))
            if args.visual and args.textual:
                emb = torch.cat([emb_v, emb_t], dim=1).numpy().reshape(-1,)
            elif args.visual:
                emb = emb_v.numpy().reshape(-1,)
            elif args.textual:
                emb = emb_t.numpy().reshape(-1,)
            fp.write(word + ' ' + ' '.join(map(str, emb)) + '\n')

if args.export:
    save_embeddings(args.save)
