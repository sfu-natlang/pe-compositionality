.data:
	cd data && ./clean.sh

.glove:
	cd glove && ./run.sh

.word2vec:
	cd word2vec && ./run.sh

.fasttext:
	cd fasttext && ./run.sh

.multimodal:
	cd multimodal && ./run.sh

.image_reco:
	cd image_reco && ./run.sh

.ocs_pcs:
	cd ocs_pcs && mkdir -p results/ && python metrics.py

all:	.data .glove .word2vec .fasttext .multimodal .image_reco .ocs_pcs
