import os
import json
from pyconll import load_from_file

train_pth = "../VnDT/VnDTv1.1-gold-POS-tags-train.conll"
test_pth = "../VnDT/VnDTv1.1-gold-POS-tags-test.conll"
dev_pth = "../VnDT/VnDTv1.1-gold-POS-tags-dev.conll"

labels = set()

def preprocess(data_path, name):
    store = []
    data = load_from_file(data_path)
    for tmp in data:
        sentence = []
        for word in tmp:
            if word.id == "1" and len(sentence) > 0:
                store.append({"words" : sentence})
                sentence = []
            sentence.append({
                "id" : word.id,
                "form" : word.form,
                "xpostag" : word.xpos,
                "deprel" : word.deprel,
                "head" : word.head
            })
            # print(word.id, word.form, word.head, word.deprel, word.xpos)
            labels.add(word.deprel)
        store.append({"words" : sentence})
    
    with open(f"../data/vndt/{name}.json", "w", encoding='utf8') as final:
            json.dump(store, final, ensure_ascii=False)

    


preprocess(train_pth, "train")
preprocess(test_pth, "test")
preprocess(dev_pth, "dev")

labels = {v : u for u, v in enumerate(labels)}
with open(f"../data/vndt/labels.json", "w") as final:
        json.dump(labels, final)