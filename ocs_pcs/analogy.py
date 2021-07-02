import torch
import glob
from metrics import load_model
import numpy as np
import re
from collections import Counter, defaultdict
from scipy.stats import chisquare, fisher_exact
import random
from scipy.spatial.distance import cosine

number_compositions   = defaultdict(dict)
is_comp_nr_significant = defaultdict(dict)

number_analogies      = defaultdict(dict)
is_analogy_nr_significant = defaultdict(dict)

def get_nn(word_emb, embeddings, id2word, K=5, verbose=True):
    nns = []
    scores = (embeddings / np.linalg.norm(embeddings, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    for i, idx in enumerate(k_best):
        nns.append( (scores[idx], id2word[idx]) )
    return nns

for K in [1,3,5,10,15]:
    compositions = defaultdict(list)
    analogies    = defaultdict(list)
            
    # Results reported in paper:
    models = [
            "../pretrained/embeddings/glove/glove.256.txt",
            "../pretrained/embeddings/word2vec/word2vec.cbow.256.vec",
            "../pretrained/embeddings/word2vec/word2vec.skip.128.vec",
            "../pretrained/embeddings/word2vec/word2vec.cbow.16.vec",
            "../pretrained/embeddings/word2vec/word2vec.skip.32.vec",
            "../pretrained/embeddings/fasttext/fasttext.skip.256.vec",
            "../pretrained/embeddings/fasttext/fasttext.skip.128.vec",
            "../pretrained/embeddings/fasttext/fasttext.cbow.128.vec",
            "../pretrained/embeddings/lms/lm.image+text.64.bi.pt.vec",
            "../pretrained/embeddings/lms/lm.image.64.bi.pt.vec",
            "../pretrained/embeddings/lms/lm.text.64.bi.pt.vec",
            "../pretrained/embeddings/image_reco.txt",
            ]

    for model in models:
        embeddings = load_model(model)
        
        id2word = list(sorted(embeddings.keys()))
        word2id = {v: k for k, v in enumerate(id2word)}

        emsize = len([embeddings[w] for w in embeddings][0])
        emb_np = np.zeros( (len(id2word), emsize) ) 
        for word in embeddings:
            emb_np[ word2id[word] ] = embeddings[word]
        embeddings = emb_np

        def get_embedding_(sign):
            if sign in word2id:
                return embeddings[word2id[sign]]
            raise KeyError(sign)

        all_compounds = [ word.split("+") for word in word2id if "+" in word  ]
        model = model.split("/")[-1]
        print(f"Checking {model} at K = {K}...")

        compositions_checked = 0
        compositions_correct = 0
        analogies_checked = 0
        analogies_correct = 0

        for i, components_1 in enumerate(all_compounds):
            compound_1 = "+".join(components_1)
            counter_1 = Counter(components_1)
            embedding_1 = get_embedding_( "+".join(components_1) )

            # COMPOSITIONALITY within compound signs:
            try:
                computed_result = np.zeros_like(embedding_1)
                formula = ""
                for sign, count in counter_1.items():
                    e = get_embedding_(sign)
                    computed_result = computed_result + count * e
                    formula += f" + {count}*{sign}" if count > 1 else f" + {sign}"
                formula = formula[3:] + f" == {compound_1}"
                neighbors = get_nn( computed_result, embeddings, id2word, K=K )
                compositions_checked += 1

                signs = counter_1.keys()
                for index, (score, sign) in enumerate(neighbors):
                    if re.sub( "_[0-9]*", "", sign ) == compound_1:
                        compositions[formula].append( (model, score, index+1, signs) )
                        compositions_correct += 1
                        break
            except KeyError:
                pass

            # ANALOGY between compound signs:
            for j, components_2 in enumerate(all_compounds):
                if j <= i:
                    continue

                compound_2 = "+".join(components_2)
                counter_2 = Counter(components_2)
                embedding_2 = get_embedding_( "+".join(components_2) )
                if any( s in components_1 and s in components_2 for s in set(components_1+components_2) ):
                    try:
                        subtract = counter_1 - counter_2
                        add = counter_2 - counter_1
                        computed_result = embedding_1
                        formula = f"{compound_1}"
                        for sign, count in subtract.items():
                            e = get_embedding_( sign )
                            computed_result = computed_result - count*e
                            formula += f" - {count}*{sign}" if count > 1 else f" - {sign}"
                        for sign, count in add.items():
                            e = get_embedding_(sign)
                            computed_result = computed_result + count*e
                            formula += f" + {count}*{sign}" if count > 1 else f" + {sign}"
                        neighbors = get_nn( computed_result, embeddings, id2word, K=K)
                        formula += f" == {compound_2}"

                        # Get a sorted list of signs involved in the alternation, 
                        # to see if there are any trends
                        alternation = [sign for sign in (counter_1-counter_2).keys()] + [sign for sign in (counter_2-counter_1).keys()]
                        alternation = sorted(list(set(alternation)))

                        analogies_checked += 1
                        for index, (score, sign) in enumerate(neighbors):
                            if re.sub( "_[0-9]*", "", sign) == compound_2:
                                analogies[formula].append( (model, score, index+1, alternation) )
                                analogies_correct += 1
                                break
                    except KeyError:
                        pass

        # Check how frequently a randomly-sampled CG appears compositional,
        # to determine whether signals in the data are significant:
        random_checked = 0
        random_correct = 0
        for idx in range(len(all_compounds)*10):
            compound = all_compounds[idx%len(all_compounds)]
            random_checked += 1
            compound = '+'.join( compound )
            id_1 = np.random.choice( range(len(word2id)) )
            id_2 = np.random.choice( range(len(word2id)) )
            computed_result = embeddings[id_1] + embeddings[id_2]
            neighbors = get_nn( computed_result, embeddings, id2word, K=K )
            for index, (score, sign) in enumerate(neighbors):
                if re.sub( "_[0-9]*$" , "", sign ) == compound:
                    random_correct += 1
        if compositions_correct > 0 or random_correct > 0:
            chi2 = fisher_exact( 
                [[compositions_checked-compositions_correct, compositions_correct],
                [random_checked-random_correct, random_correct]],
                alternative='less'
                )
            number_compositions[model][K] = compositions_correct
            is_comp_nr_significant[model][K] = chi2[1]

        random_analogies_checked = 0
        random_analogies_correct = 0
        for idx in range(len(all_compounds)*10):
            try:
                # construct a random analogy of the form a:b::c:d
                c_1 = random.choice( all_compounds )
                c_1 = '+'.join(c_1)
                id_2 = np.random.choice( range(len(word2id)) )
                id_3 = np.random.choice( range(len(word2id)) )
                c_2 = random.choice( all_compounds )
                c_2 = '+'.join(c_2)
                target = c_2
                computed_result = get_embedding_( c_1 ) - embeddings[id_2] + embeddings[id_3]
                neighbors = get_nn( computed_result, embeddings, id2word, K=K)
                random_analogies_checked += 1
                for index, (score, sign) in enumerate(neighbors):
                    if re.sub( "_[0-9]*$" , "", sign ) == target:
                        random_analogies_correct += 1
            except:
                pass
        if analogies_correct > 0 or random_analogies_correct > 0:
            chi2 = fisher_exact( 
                [[analogies_checked-analogies_correct, analogies_correct],
                [random_analogies_checked-random_analogies_correct, random_analogies_correct]],
                alternative='less'
                )
            number_analogies[model][K] = analogies_correct
            is_analogy_nr_significant[model][K] = chi2[1]
    
    sign_counts = defaultdict(int)
    with open(f"csvs/composition.{K}.csv", "w") as fp:
        fp.write("formula, number of models where formula holds, list of models where formula holds, avg. similarity to target word, avg. rank of target word\n")
        for formula, details in compositions.items():
            models  = [detail[0] for detail in details]
            scores  = [detail[1] for detail in details]
            indices = [detail[2] for detail in details]
            fp.write("%s,%d,%s,%s,%s\n"%(
                formula,
                len(models),
                ';'.join(models),
                np.mean(scores),
                np.mean(indices)
                ))
    
            signs = [detail[3] for detail in details][0]
            for sign in signs:
                sign_counts[sign] += 1
    
        fp.write("How often does each sign occur?\n")
        for sign, count in sorted(sign_counts.items(), key=lambda x:x[-1]):
            fp.write("%s,%d\n"%(sign, count))
    
    alternation_counts = defaultdict(int)
    with open(f"csvs/analogy.{K}.csv", "w") as fp:
        fp.write("formula, number of models where formula holds, list of models where formula holds, avg. similarity to target word, avg. rank of target word\n")
        for formula, details in analogies.items():
            models  = [detail[0] for detail in details]
            scores  = [detail[1] for detail in details]
            indices = [detail[2] for detail in details]
            fp.write("%s,%d,%s,%s,%s\n"%(
                formula,
                len(models),
                ';'.join(models),
                np.mean(scores),
                np.mean(indices)
                ))
        
            alternation = tuple([detail[3] for detail in details][0])
            alternation_counts[alternation] += 1
    
        fp.write("Count of each alternation observed in the analogies:\n")
        for alternation, count in sorted(alternation_counts.items(), key=lambda x:x[1]):
            fp.write("%s,%d\n"%(alternation, count))

print()
print("Composition")
print("model & 1 & 3 & 5 & 10 & 15 \\\\")
for model in is_comp_nr_significant:
    print(model, end=' & ')
    print(' & '.join(["%d%s"%(n,"" if is_comp_nr_significant[model][k] > 0.05 else "*") for k, n in number_compositions[model].items()]), end='\\\\\n')

print()
print("Analogy")
print("model & 1 & 3 & 5 & 10 & 15 \\\\")
for model in is_analogy_nr_significant:
    print(model, end=' & ')
    print(' & '.join(["%d%s"%(n,"" if is_analogy_nr_significant[model][k] > 0.05 else "*") for k, n in number_analogies[model].items()]), end='\\\\\n')
