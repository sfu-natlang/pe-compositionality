from gensim.models import Word2Vec

for v in ["", "no"]:
    for amt in ["more", "less"]:
        SETTING = f"{v}variants.prune_{amt}"
        for dim in [16, 32, 64, 128, 256]:
            print(SETTING, dim)
            with open(f"../data/{SETTING}.txt") as fp:
                data = fp.read().split("\n")[:-1]
                data = [ [word for word in line.split(" ") if word != ""] for line in data]
                for sg, name in enumerate(["cbow", "skip"]):
                    model = Word2Vec(sentences=data, sg=sg, vector_size=dim, window=15, min_count=0, workers=4, epochs=3000)
                    model.wv.save_word2vec_format(f"output/word2vec.{name}.{SETTING}.{dim}.vec")
