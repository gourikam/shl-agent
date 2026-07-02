"""
Catalog retrieval: hybrid structured filter + keyword scoring (BM25-ish).
Deliberately stdlib-only — no vector DB, no heavy ML deps. The catalog is a
few hundred to ~1-2k short text records; exact term/product-name matching
matters more here than semantic embedding similarity, and this keeps cold
starts on free hosting fast and dependency-light.
"""
import re
import math
from collections import Counter
from typing import List, Dict, Optional

TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class CatalogRetriever:
    def __init__(self, catalog: List[Dict]):
        """
        catalog: list of dicts with at least:
            name, url, description, keys (list[str]), job_levels (list[str]),
            languages (list[str]), duration (str)
        """
        self.catalog = catalog
        self.doc_tokens = []
        self.doc_freq = Counter()
        for item in catalog:
            text = f"{item.get('name','')} {item.get('description','')}"
            toks = tokenize(text)
            self.doc_tokens.append(toks)
            for t in set(toks):
                self.doc_freq[t] += 1

        self.N = len(catalog)
        self.avgdl = sum(len(t) for t in self.doc_tokens) / max(self.N, 1)
        self.k1 = 1.5
        self.b = 0.75

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        doc = self.doc_tokens[doc_idx]
        dl = len(doc)
        tf = Counter(doc)
        score = 0.0
        for qt in query_tokens:
            n_qt = tf.get(qt, 0)
            if n_qt == 0:
                continue
            df = self.doc_freq.get(qt, 0)
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            denom = n_qt + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            score += idf * (n_qt * (self.k1 + 1)) / max(denom, 1e-9)
        return score

    def search(
        self,
        query: str,
        top_k: int = 20,
        job_level: Optional[str] = None,
        language: Optional[str] = None,
        test_type: Optional[str] = None,
    ) -> List[Dict]:
        q_tokens = tokenize(query)
        scored = []
        for i, item in enumerate(self.catalog):
            if job_level and job_level not in item.get("job_levels", []):
                continue
            if language and item.get("languages") and language not in item.get("languages", []):
                continue
            if test_type and test_type not in item.get("keys", []):
                continue
            s = self._bm25_score(q_tokens, i)
            scored.append((s, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        # If filters wiped out everything, fall back to unfiltered search.
        if not scored:
            return self.search(query, top_k=top_k)
        return [item for _, item in scored[:top_k]]

    def get_by_name(self, name: str) -> Optional[Dict]:
        name_l = name.lower().strip()

        # 1. exact match
        for item in self.catalog:
            if item.get("name", "").lower().strip() == name_l:
                return item

        # 2. substring match (either direction) — catches minor punctuation/spacing diffs
        for item in self.catalog:
            item_l = item.get("name", "").lower().strip()
            if name_l in item_l or item_l in name_l:
                return item

        # 3. token-overlap (Jaccard) fuzzy match — catches reordered/partial names
        #    e.g. "Microsoft Excel 365 (New)" vs "Microsoft Excel 365 - Essentials (New)"
        query_tokens = set(tokenize(name_l))
        if not query_tokens:
            return None
        best_item, best_score = None, 0.0
        for item in self.catalog:
            item_tokens = set(tokenize(item.get("name", "")))
            if not item_tokens:
                continue
            overlap = len(query_tokens & item_tokens)
            union = len(query_tokens | item_tokens)
            jaccard = overlap / union if union else 0.0
            # require the query to be substantially covered, not just any overlap
            coverage = overlap / len(query_tokens)
            score = jaccard * 0.5 + coverage * 0.5
            if score > best_score:
                best_score, best_item = score, item
        # threshold: require decent confidence before accepting a fuzzy match,
        # to avoid silently grounding a hallucinated name to the wrong product.
        if best_score >= 0.6:
            return best_item
        return None

    def get_many_by_names(self, names: List[str]) -> List[Dict]:
        out = []
        for n in names:
            item = self.get_by_name(n)
            if item:
                out.append(item)
        return out
