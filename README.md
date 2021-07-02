Code, models, and data for _Compositionality of Complex Graphemes in the Undeciphered Proto-Elamite Script using Image and Text Embedding Models_, published in Findings of ACL 2021.

## Models and Training

To build all models from scratch and generate the results from the paper, run 
```bash
make all
```
from the root directory.

Alternatively, each model has its own directory with a `run.sh` file which will train all versions of that model used in the paper. You must generate the input files with `make .data` before training any models.

Pretrained models are included in `pretrained/models`.

## Pretrained Embeddings
Embeddings from all models used in the paper are included in `pretrained/embeddings`.

## Statistics and Evaluations
All statistics and analysis scripts are located in `ocs\_pcs`. 

```bash
python metrics.py && python stats.py
```
will compute PCS for every sign in every model and summarize the resulting scores.

```bash
python analogy.py
```
computes the number of compositional signs and analogies in each model and outputs the results cited in the paper. For each value of _k_, a csv file will be saved to `ocs\_pcs/csvs` listing information about which signs are compositional, to what degree they are compositional, and in which models.

These scripts use the pretrained embeddings included with the repository, so they can be run without retraining the models from scratch.
