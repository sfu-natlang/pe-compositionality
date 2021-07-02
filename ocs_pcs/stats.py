from collections import defaultdict
import os
import numpy as np

from scipy.stats import mannwhitneyu, ttest_1samp

P_THRESHOLD = 0.05

totals  = defaultdict(lambda:{'inside': {'ocs':[], 'pcs': []}, 'outside': {'ocs':[], 'pcs': []}})
raw     = defaultdict(lambda:defaultdict(lambda:{'inside': {'ocs':[], 'pcs': []}, 'outside': {'ocs':[], 'pcs': []}}))

models = []

for filename in os.listdir("results"):
    models.append(filename)
    with open(os.path.join("results", filename)) as fp:
        lines = fp.read().split("\n")
        

        for line in lines:
            if line.strip() == "":
                continue
            word, ocs, pcs, ocs_raw, pcs_raw, n_pairs = line.split(",")
            word, location = word.split("_")
            n_pairs = int(n_pairs)
            totals[filename][location]['ocs'].append( float(ocs) )
            totals[filename][location]['pcs'].append( float(pcs) )

            # For t-tests, we want large(ish) samples so that the means
            # are closer to normal. If there are too few pairs, the 
            # effective sample size is reduced, since some shuffles 
            # will be seen twice. There are n_pairs**2 possible results 
            # of shuffling, of which n_pairs are avoided because they
            # are the true pairings. Thus there are n_pairs**2 - n_pairs
            # possible shuffles that we could sample. 
            if n_pairs**2 - n_pairs > 10:
                ocs_raw = list(map( float, ocs_raw.split(";") ))
                pcs_raw = list(map( float, pcs_raw.split(";") ))
                raw[word][filename][location]['ocs'] =  ocs_raw
                raw[word][filename][location]['pcs'] =  pcs_raw

pcs_by_sign = []
for word in raw:
    for location in ['inside', 'outside']:
        results = []
        for model in raw[word]:
            values = raw[word][model][location]['pcs']
            ttest = ttest_1samp( values, 0.5 )
            if ttest.pvalue < P_THRESHOLD:
                pcs_by_sign.append( (word, location, model, np.mean(values)) )

print("="*80)
print("Signs with PCS significantly different from 0.5, grouped by model:")
for model in models:
    print("\t",model)
    printed_something = False
    for word, location, m, pcs in sorted( pcs_by_sign, key=lambda x:x[-1] ):
        if m == model:
            print(f"\t{word}\t{location}\t{pcs}")
            printed_something = True
    if not printed_something:
        print("\t(no signs have PCS significantly different from 0.5)")
    print()

print("="*80)
print("PCS averaged across all models:")
averages = []
for word in raw:
    for location in ['inside', 'outside']:
        all_pcs = []
        for model in raw[word]:
            all_pcs += raw[word][model][location]['pcs']
        if len(all_pcs) > 0:
            averages.append((word, location, np.mean(all_pcs)))
for word, location, mean in sorted(averages, key=lambda x:x[-1]):
    print(f"\t{word}\t{location}\t{mean}")
print()

print("="*80)
print("PCS averaged across non-visual models:")
averages = []
for word in raw:
    for location in ['inside', 'outside']:
        all_pcs = []
        for model in raw[word]:
            if "image" in model:
                continue
            all_pcs += raw[word][model][location]['pcs']
        if len(all_pcs) > 0:
            averages.append((word, location, np.mean(all_pcs)))
for word, location, mean in sorted(averages, key=lambda x:x[-1]):
    print(f"\t{word}\t{location}\t{mean}")
print()

print("="*80)
print("PCS averaged across all relations:")
print("& \multicolumn{2}{c}{OCS} & \multicolumn{2}{c}{PCS} \\\\")
print("model & inside & outside & inside & outside\\\\\\hline")
for model in totals:
    print(model, end='')
    for metric in ['pcs']:
        for location in ['inside', 'outside']:
            print("&","%.03f"%(np.mean(totals[model][location][metric])),end='' )
            inside_data  = totals[model]['inside'][metric]
            outside_data = totals[model]['outside'][metric]
            if len(inside_data) > 10 and len(outside_data) > 10:
                mwu = mannwhitneyu( inside_data, outside_data, alternative='two-sided' )
                if mwu.pvalue < P_THRESHOLD:
                    print("*", end='')
    print('\\\\')

print("="*80)
print("OCS and PCS averaged across all relations and models:")
for metric in ['ocs', 'pcs']:
    cross_model_inside  = sum([totals[model]['inside'][metric] for model in totals],[])
    cross_model_outside = sum([totals[model]['outside'][metric] for model in totals],[])
    print(np.mean(cross_model_inside))
    print(np.mean(cross_model_outside))
    if len(cross_model_inside) > 20 and len(cross_model_outside) > 20:
        mwu = mannwhitneyu( cross_model_inside, cross_model_outside, alternative='two-sided' )
        if mwu.pvalue < P_THRESHOLD:
            print(f"* across models, {metric} has a significant interaction with location")
            print(f"* {metric} is significantly larger for", 
                    "inner" if np.mean(cross_model_inside) > np.mean(cross_model_outside) else "outer",
                    "signs")
            print(mwu)
    print()
