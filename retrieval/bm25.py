import json, string, unicodedata
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
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = "".join(" " if c in string.punctuation else c for c in s)
    return " ".join(s.split())

def has_answer(text, answers):
    t = f" {normalize(text)} "
    return any(f" {normalize(a)} " in t for a in answers if normalize(a))

def retrieve_unanswerable(query, answers, k=5, pool=100):
    hits = searcher.search(query, k=pool)
    out = [(h.docid, h.score) for h in hits
           if not has_answer(get_text(h.docid), answers)][:k]
    return out