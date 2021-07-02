"""
Adapted from 
https://github.com/bootphon/measuring-regularities-in-word-embeddings/blob/master/metrics.py
"""
import glob

import numpy as np
import sklearn

import sys
import time

from os.path import exists
from os import mkdir

from sklearn.metrics.pairwise import cosine_similarity as cos_sim

import torch
import json
import os

import re

def load_model(path):
    vectors = dict()
    with open(path) as fp:
        lines = fp.read().split("\n")
        for line in lines:
            if line != "":
                line = line.strip().split(" ")
                if len(line) == 2:
                    continue
                word = line[0].replace("*", "").replace("^", "")
                values = list(map(float,line[1:]))
                values = np.array(values)
                vectors[word] = values
    return vectors

def clean_pairs(model, names, pair_sets, name):
    cleaned_names = []
    cleaned_pair_sets = []
    for i, pair_set in enumerate( pair_sets ):
        cleaned_pair_set = []
        for pair in pair_set:
            for word in pair:
                if word not in model:
                    alternatives = [w for w in model if re.sub( "-[a-zA-Z0-9]*", "", word ) in w and "+" not in w]
                    print(f"WARN: {word} is OOV in {name}. Possible fallbacks: {alternatives}.")
                    break
            else:
                cleaned_pair_set.append( pair )
        if len( cleaned_pair_set ) >= 3: # TODO
            if any( cleaned_pair_set[i][1] == cleaned_pair_set[j][1] for i in range(len(cleaned_pair_set)) for j in range(i+1,len(cleaned_pair_set)) ):
                print("WARN: duplicate word detected in", cleaned_pair_set)
                continue
            cleaned_names.append( names[i] )
            cleaned_pair_sets.append( cleaned_pair_set )
    return cleaned_names, cleaned_pair_sets

def get_names_pairs(model):
    all_compounds = [ word for word in model if word.count("+") == 1 ]
    all_compounds = [ re.sub( "_[0-9]*$", "", word ) for word in all_compounds ]
    first_components = list(set([ compound.split("+")[0] for compound in all_compounds ]))
    last_components  = list(set([ compound.split("+")[-1] for compound in all_compounds ]))

    names = []
    pairs = []
    for component in first_components:
        names.append( component + "_outside" )
        pairs.append( [] )
        for compound in all_compounds:
            outside, inside = compound.split("+")
            if outside == component:
                pairs[-1].append( ("%s"%(inside), "%s"%(compound)) )
    for component in last_components:
        names.append( component + "_inside" )
        pairs.append( [] )
        for compound in all_compounds:
            outside, inside = compound.split("+")
            if inside == component:
                pairs[-1].append( ("%s"%(outside), "%s"%(compound)) )
    return names, pairs

def token_embedding(tokenizer, model, word):
    tokenized_text = tokenizer.tokenize(word)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    embeds = np.array([model[i] for i in indexed_tokens])
    embed = np.mean(embeds, axis=0)
    return(embed)

def permutation_onecycle(n):
    if type(n) == tuple:
        n1, n2 = n[0], n[1]
    else:
        n1, n2 = 0, n
    l=np.random.permutation(range(n1, n2))
    while any(l[i-n1] == i for i in range(n1, n2)):
        l=np.random.permutation(range(n1, n2))
    # Original: prone to hang with an infinite loop:
    #for i in range(n1, n2):
        #if i==l[i-n1]:
            #j=np.random.randint(n1, n2)
            #while j==l[j-n1]:
                #j=np.random.randint(n1, n2)
            #l[i-n1], l[j-n1] = l[j-n1], l[i-n1]
    return(l)

def permutation_onecycle_avoidtrue(n, real): #May be a more optimal way
    test = False
    perm = permutation_onecycle(n)
    for i_r in range(len(real)):
        if real[i_r][1] == real[perm[i_r]][1]:
            test = True
    while test:
        test = False
        perm = permutation_onecycle(n)
        for i_r in range(len(real)):
            if real[i_r][1] == real[perm[i_r]][1]:
                test = True
    return(perm)

def shuffled_directions(model, idx_start, idx_end):
    perm_list = permutation_onecycle(len(idx_start))
    dirs = np.array([[model.get_vector(idx_end[perm_list[i]]) - model.get_vector(idx_start[i])
                                          for i in range(len(idx_start))]])
    return(dirs)

def similarite_offsets(list_offsets):
    sim_offsets = []
    for i in range(len(list_offsets)):
        sim_offsets.append([])
        list_tuples = list(list_offsets[i])
        for j in range(len(list_tuples)):
            for k in range(j+1,len(list_tuples)):
                sim_offsets[-1].append(cos_sim([list_tuples[j]], [list_tuples[k]])[0][0])
    return(np.array(sim_offsets, dtype=object))

def OCS_PCS(nb_perm, similarities, similarities_shuffle):
    ocs, pcs = [], []
    ocs_raw, pcs_raw = [], []
    print('# Computing the OCS and PCS metrics')
    for i in range(len(similarities)):
        pcs_list = []
        for perm in range(nb_perm):
            y_true = [1 for j in range(len(similarities[i]))]+[0 for j in range(len(similarities_shuffle[perm][i]))]
            y_scores = list(similarities[i])+list(similarities_shuffle[perm][i])
            auc_temp = sklearn.metrics.roc_auc_score(y_true,y_scores)
            pcs_list.append(auc_temp)
        pcs.append(np.mean(pcs_list))
        ocs.append(np.mean(similarities[i]))
        pcs_raw.append(pcs_list)
        ocs_raw.append(similarities[i])
    print('# Computed the OCS and PCS metrics')
    return(ocs, pcs, ocs_raw, pcs_raw)

def word_embedding(model, word):
    return model[word]

def context_sentence(name):
    with open(os.path.join('BATS_3.0','context_sentences.json')) as json_file:
        context_sentences = json.load(json_file)
    return(context_sentences[name[:3]])

def sublist(liste, pattern):
    indx = -1
    for i in range(len(liste)):
        if liste[i] == pattern[0] and liste[i:i+len(pattern)] == pattern:
           indx = i
    return indx

def offset_contextual(model, tokenizer, model_name, name, w1, w2):
    context = context_sentence(name)
    c1, c2 = context

    sentence = ' '.join([c1, w1, c2, w2])
    if model_name == 'gpt-context':
        sentence = "[CLS] " + sentence + " [SEP]"
    else:
        w1 = " "+w1
        w2 = " "+w2

    tokenized_sentence = tokenizer.tokenize(sentence)
    tokenized_w1 = tokenizer.tokenize(w1)
    tokenized_w2 = tokenizer.tokenize(w2)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        if model_name == 'gpt-context':
            segments_ids = [1] * len(tokenized_sentence)
            segments_tensors = torch.tensor([segments_ids])
            outputs = model(tokens_tensor, segments_tensors)
        else:
            outputs = model(tokens_tensor)
        hidden_states = outputs[2]

    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)

    token_vecs = []
    for token in token_embeddings:
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        token_vecs.append(cat_vec)

    idx_w1 = sublist(tokenized_sentence, tokenized_w1)
    idx_w2 = sublist(tokenized_sentence, tokenized_w2)
    len_w1 = len(tokenized_w1)
    len_w2 = len(tokenized_w2)

    embd_w1 = torch.mean(torch.stack(token_vecs[idx_w1:idx_w1 + len_w1 + 1]), dim=0)
    embd_w2 = torch.mean(torch.stack(token_vecs[idx_w2:idx_w2 + len_w2 + 1]), dim=0)

    return(embd_w2 - embd_w1)


def offset(model, w1, w2, name):
    return (word_embedding(model, w2) - \
            word_embedding(model, w1))


def offsets(model, pairs_sets, names=None):
    return np.array([[offset(model, i[0], i[1], names[k])
                       for i in pairs_sets[k]]
                       for k in range(len(pairs_sets))], dtype=object)

def shuffled_offsets(model, pairs_sets, nb_perms=50, avoid_true=True, names=None):
    shf_offsets = []
    for k in range(len(pairs_sets)):
        shf_offsets.append([])
        for perm in range(nb_perms):
            if avoid_true:
                perm_list = permutation_onecycle_avoidtrue(len(pairs_sets[k]), pairs_sets[k])
            else:
                perm_list = permutation_onecycle(len(pairs_sets[k]))
            offs = [offset(model, pairs_sets[k][i][0], pairs_sets[k][perm_list[i]][1], names[k])
                    for i in range(len(pairs_sets[k]))]
            shf_offsets[-1].append(offs)
    return (shf_offsets)

def normal_and_shuffled_offsets(model, pairs_sets, nb_perms=50, names=None):
    print('# Computing the normal and shuffled offsets')

    normal_offsets = offsets(model, pairs_sets, names=names)
    shf_offsets = shuffled_offsets(model, pairs_sets, nb_perms=nb_perms, names=names)
    print('# Computed the normal and shuffled offsets')
    return(normal_offsets, shf_offsets)


def metrics_from_model(model, name, nb_perms=50):
    names, pairs_sets = get_names_pairs(model)
    names, pairs_sets = clean_pairs(model, names, pairs_sets, name)

    normal_offsets, shf_offsets = normal_and_shuffled_offsets(model, pairs_sets, nb_perms=nb_perms, names=names)

    print('# Computing the similarities of the normal and shuffled offsets')
    similarities = similarite_offsets(normal_offsets)
    similarities_shuffle = [similarite_offsets(np.array(shf_offsets, dtype=object)[:, perm])
                            for perm in range(nb_perms)]
    print('# Computed the similarities of the normal and shuffled offsets')

    ocs, pcs, ocs_raw, pcs_raw = OCS_PCS(nb_perms, similarities, similarities_shuffle)

    return (names, pairs_sets, ocs, pcs, ocs_raw, pcs_raw)

if __name__ == "__main__":

    nb_perms = 50
    models = [
            "../pretrained/embeddings/glove/glove.64.txt",
            "../pretrained/embeddings/word2vec/word2vec.cbow.64.vec",
            "../pretrained/embeddings/word2vec/word2vec.skip.64.vec",
            "../pretrained/embeddings/fasttext/fasttext.skip.64.vec",
            "../pretrained/embeddings/fasttext/fasttext.cbow.64.vec",
            "../pretrained/embeddings/lms/lm.image+text.64.bi.pt.vec",
            "../pretrained/embeddings/lms/lm.image.64.bi.pt.vec",
            "../pretrained/embeddings/lms/lm.text.64.bi.pt.vec",
            "../pretrained/embeddings/image_reco.txt",
            ]
    for name in models:
        model = load_model(name)
        names, pairs, ocs, pcs, ocs_raw, pcs_raw = metrics_from_model(model, name, nb_perms=nb_perms)
        print("# Sucessfully computed the OCS and PCS metrics from", str(name))

        filename = name.split("/")[-1]
        with open(f"results/{filename}.ocs_pcs", "w") as fp:
            for i in range(len(names)):
                fp.write("%s,%f,%f,%s,%s,%d\n"%(names[i], ocs[i], pcs[i], 
                    ';'.join(map(str,ocs_raw[i])), ';'.join(map(str,pcs_raw[i])), len(pairs[i])))
