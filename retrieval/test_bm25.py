import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from pyserini.search.lucene import LuceneSearcher

search = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')

def retrieve(query: str, k: int, exclude: set = ()) -> list[tuple[str, float, str]]:
    exclude = set(exclude)
    hits = search.search(query, k=k + len(exclude))
    hits = [hit for hit in hits if hit.docid not in exclude][:k]
    return [(hit.docid, hit.score, json.loads(search.doc(hit.docid).raw())["contents"]) for hit in hits]

for q in ["who wrote the origin of species",
          "On the Origin of Species author Charles Darwin"]:
    print("\n问题:", q)
    for docid, score, text in retrieve(q, k=10):
        print(docid, round(score, 2), text[:80].replace("\n", " "))