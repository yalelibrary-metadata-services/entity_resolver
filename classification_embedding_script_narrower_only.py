#!/usr/bin/env python
"""
Narrower-only Knowledge Classification Embedding Script

Embeds only narrower SKOS concepts with full broader-context, indexes them,
and adds concept-overlap diagnostics plus a classification_indicator flag.
"""
import os, sys, logging, json, pickle, time, argparse, csv, hashlib
from typing import Dict, List, Any
from enum import Enum
import numpy as np, yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

# import core classes from the averaging script
sys.path.append(os.path.dirname(__file__))
from classification_embedding_script_with_averaging import (
    SKOSTaxonomyProcessor,
    ClassificationEmbedding,
    ClassificationMode,
    ClassificationResult
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NarrowerClassificationEmbedding(ClassificationEmbedding):
    def generate_embeddings(self, classification_scheme: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        logger.info("Generating embeddings for narrower-only concepts")
        items = []
        # select only narrower concepts (path length > 1)
        for uri, terms in classification_scheme.items():
            path = self.taxonomy_processor.get_concept_path(uri)
            if len(path) < 2:
                continue
            # build text: all ancestors then this concept
            parts = []
            for node in path:
                lbl = self.taxonomy_processor.get_concept_label(node)
                alt = self.taxonomy_processor.concept_alt_labels.get(node, [])
                dfn = self.taxonomy_processor.concept_definitions.get(node, "")
                sc = self.taxonomy_processor.concept_scope_notes.get(node, "")
                txt = lbl
                if alt:
                    txt += f" (alt: {', '.join(alt)})"
                if dfn:
                    txt += f": {dfn}"
                if sc:
                    txt += f" [{sc}]"
                parts.append(txt)
            emb_text = ". ".join(parts)
            pref = terms[0]
            items.append((uri, pref, emb_text))
        # batch embed exactly as parent does
        # monkey-patch classification_scheme to items for parent
        # reuse low-level batch logic
        return self._batch_embed_items(items)

    def _batch_embed_items(self, items: List[tuple]) -> Dict[str, Any]:
        # copy of parent logic to process items list
        # rate-limit and thread logic from parent
        lock = threading.Lock()
        self.tokens_this_minute = 0
        self.requests_this_minute = 0
        self.minute_start = time.time()
        batches = [items[i:i+self.batch_size] for i in range(0, len(items), self.batch_size)]
        embedding_map = {}
        total_tokens = 0
        max_workers = min(__import__('multiprocessing').cpu_count(), self.config.get('embedding_workers', 4))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._process_embedding_batch, batch, lock): batch for batch in batches}
            for f in tqdm(as_completed(futures), total=len(batches), desc="Generating embeddings", unit="batch"):
                em, tk = f.result()
                embedding_map.update(em)
                total_tokens += tk
        logger.info(f"Generated {len(embedding_map)} embeddings using {total_tokens} tokens")
        self.concept_embedding_mapping = embedding_map
        return embedding_map

    def compare_person_ids(self, person_id1: str, person_id2: str, mode: ClassificationMode = None) -> Dict[str, Any]:
        res = super().compare_person_ids(person_id1, person_id2, mode)
        # ensure vector_similarity is JSON serializable
        if 'vector_similarity' in res:
            res['vector_similarity'] = float(res['vector_similarity'])
        # detect overlap in top_k concepts
        key1 = f"{self.person_id_mappings[person_id1]['composite_hash']}_{self.top_k}_{res['mode']}"
        key2 = f"{self.person_id_mappings[person_id2]['composite_hash']}_{self.top_k}_{res['mode']}"
        cls1 = self.classification_cache.get(key1)
        cls2 = self.classification_cache.get(key2)
        # require each entity's top concept to appear in the other's top-3
        top1_1 = cls1.top_concepts[0]['concept_uri'] if cls1 and cls1.top_concepts else None
        top1_2 = cls2.top_concepts[0]['concept_uri'] if cls2 and cls2.top_concepts else None
        top3_1 = {c['concept_uri'] for c in (cls1.top_concepts[:3] if cls1 else [])}
        top3_2 = {c['concept_uri'] for c in (cls2.top_concepts[:3] if cls2 else [])}
        overlap = []
        if top1_1 and top1_1 in top3_2:
            overlap.append(top1_1)
        if top1_2 and top1_2 in top3_1:
            overlap.append(top1_2)
        # indicator is 1 only if both top concepts overlap reciprocally
        indicator = 1 if len(overlap) == 2 else 0
        logger.info(f"Concept overlap between {person_id1} and {person_id2}: {overlap}")
        res['concept_overlap'] = list(overlap)
        res['classification_indicator'] = indicator

        # Print embedding strings for each top concept of both entities
        def embedding_string_for_concept(concept_uri):
            path = self.taxonomy_processor.get_concept_path(concept_uri)
            parts = []
            for node in path:
                lbl = self.taxonomy_processor.get_concept_label(node)
                alt = self.taxonomy_processor.concept_alt_labels.get(node, [])
                dfn = self.taxonomy_processor.concept_definitions.get(node, "")
                sc = self.taxonomy_processor.concept_scope_notes.get(node, "")
                txt = lbl
                if alt:
                    txt += f" (alt: {', '.join(alt)})"
                if dfn:
                    txt += f": {dfn}"
                if sc:
                    txt += f" [{sc}]"
                parts.append(txt)
            return ". ".join(parts)
        print("\n=== Top concepts for", person_id1, "===")
        if cls1:
            for i, c in enumerate(cls1.top_concepts[:self.top_k]):
                print(f"Rank {i+1}: {c['concept_uri']} | {embedding_string_for_concept(c['concept_uri'])}")
        print("\n=== Top concepts for", person_id2, "===")
        if cls2:
            for i, c in enumerate(cls2.top_concepts[:self.top_k]):
                print(f"Rank {i+1}: {c['concept_uri']} | {embedding_string_for_concept(c['concept_uri'])}")
        return res

    def update_training_dataset(self) -> Dict[str, Any]:
        metrics = super().update_training_dataset()
        out = metrics.get('output_path')
        if out and os.path.exists(out):
            tmp = out + '.tmp'
            with open(out, 'r', newline='', encoding='utf-8') as inf, \
                 open(tmp, 'w', newline='', encoding='utf-8') as outf:
                dr = csv.DictReader(inf)
                flds = dr.fieldnames + ['classification_indicator']
                wr = csv.DictWriter(outf, fieldnames=flds, delimiter=dr.reader.dialect.delimiter)
                wr.writeheader()
                for r in dr:
                    # 1 if classification_label not 'unknown'
                    r['classification_indicator'] = '1' if r.get('classification_label','unknown') != 'unknown' else '0'
                    wr.writerow(r)
            os.replace(tmp, out)
            logger.info(f"Added classification_indicator to {out}")
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--diagnostic', nargs=2)
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    cfg = yaml.safe_load(open(args.config))
    emb = NarrowerClassificationEmbedding(cfg, reset_mode=args.reset)
    scheme = emb.load_classification_scheme()
    if args.reset:
        emb.generate_embeddings(scheme)
        emb.save_embeddings()
        emb.index_embeddings(scheme)
    else:
        emb.load_embeddings()
    metrics = emb.update_training_dataset()
    if args.diagnostic:
        out = emb.compare_person_ids(args.diagnostic[0], args.diagnostic[1])
        print(json.dumps(out, indent=2))
    emb.close()

if __name__ == '__main__':
    main()
