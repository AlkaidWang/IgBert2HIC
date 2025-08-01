from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from sklearn.metrics import r2_score
import pandas as pd
import numpy
import torch

tokeniser = BertTokenizer.from_pretrained("./", do_lower_case=False)
model = BertForSequenceClassification.from_pretrained('Tm-1/', num_labels=1)

test_set = pd.read_csv("../Tm.txt",sep="\t").dropna().drop_duplicates()
#test_set = pd.read_csv("Tm-1.txt",sep="\t").dropna().drop_duplicates()

test_set["SEQ"]=test_set["VH"].apply(lambda x: ' '.join(x)) + ' [SEP] '+ test_set["VL"].apply(lambda x: ' '.join(x))
print(f'{len(test_set)} testing set records')

tokens = tokeniser.batch_encode_plus(
    test_set["SEQ"], 
    add_special_tokens=True, 
    pad_to_max_length=True, 
    return_tensors="pt",
    return_special_tokens_mask=True
)

output = model(
    input_ids=tokens['input_ids'], 
    attention_mask=tokens['attention_mask']
)

logits = output.logits.detach().cpu().numpy()
r2 = r2_score(test_set['Tm'],logits)
print("  R2: {0:.2f}".format(r2))

test_set['PRE']=logits
print(test_set)
test_set.to_csv('test_Tm-1.csv', index=False)
