import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset

nq = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq", split="test")
print("Number of questions:", len(nq))
print("column names:", nq.column_names)
for i in range(3):
    print(nq[i])
