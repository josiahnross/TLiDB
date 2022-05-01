from transformers import BertForMaskedLM

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
cls = model.cls
print(cls)