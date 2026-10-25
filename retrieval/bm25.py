import json, string
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr-100w")

def get_text(docid):
    return json.loads(searcher.doc(docid).raw())["contents"]

def retrieve(query, k=5, exclude=()):
    exclude = set(exclude)
    hits = searcher.search(query, k=k + len(exclude))
    hits = [h for h in hits if h.docid not in exclude][:k]
    return [(h.docid, h.score) for h in hits]

def normalize(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def has_answer(text, answers):
    t = normalize(text)
    return any(normalize(a) in t for a in answers)
