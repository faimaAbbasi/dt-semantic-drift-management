import re
from rdflib import Graph, URIRef, RDFS, RDF, OWL, Literal, BNode, Namespace
from typing import List, Tuple, Set, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
import csv
import json
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from typing import Set, Dict, List, Tuple
import numpy as np
import jellyfish # type: ignore
from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import xml.etree.ElementTree as ET
import subprocess
import re
import json
import csv
from typing import List, Dict
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


TARGET_OWL_PATH =  'C:\\Users\\abbasi\\Desktop\\drift-mgt-mddt\\ontology-layer\\brick.ttl'
HOPS_FOR_CONTEXT = 7

# ------------------ UTILITIES ------------------

def normalize_uri(uri: str) -> str:
    return uri.strip().strip('<>').rstrip('/')

def is_uri(value: str) -> bool:
    return bool(re.match(r'^(http|https|ftp|urn):', value))

def add_triple_safe(graph: Graph, s: str, p: str, o: str):
    if not is_uri(s) or not is_uri(p):
        return
    s_ref = URIRef(s)
    p_ref = URIRef(p)
    o_ref = URIRef(o) if is_uri(o) else Literal(o.strip() if isinstance(o, str) else o)
    graph.add((s_ref, p_ref, o_ref))

def extract_all_triples(graph: Graph) -> List[Tuple[str, str, str]]:
    return [(str(s), str(p), str(o)) for s, p, o in graph if not isinstance(s, BNode) and not isinstance(o, BNode)]

def extract_local_name(uri: str) -> str:
    """
    Extract local name from URI (after last # or /)
    """
    if '#' in uri:
        return uri.split('#')[-1]
    else:
        return uri.split('/')[-1]
    
def split_camel_case(text: str) -> str:
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

# ------------------ KNOWLEDGE GRAPH ------------------


def extract_classes(graph: Graph) -> set[str]:
    classes = set()
    for c in graph.subjects(RDF.type, OWL.Class):
        if isinstance(c, URIRef):
            classes.add(str(c))
    for c in graph.subjects(RDF.type, RDFS.Class):
        if isinstance(c, URIRef):
            classes.add(str(c))
    return classes

def extract_subclass_triples(graph: Graph) -> List[Tuple[str, str, str]]:
    triples = []
    for subclass, _, superclass in graph.triples((None, RDFS.subClassOf, None)):
        if isinstance(subclass, URIRef) and isinstance(superclass, URIRef):
            triples.append((str(subclass), str(RDFS.subClassOf), str(superclass)))
    return triples

def extract_properties(graph: Graph) -> Set[str]:
    props = set()
    for p in graph.subjects(RDF.type, OWL.ObjectProperty):
        if isinstance(p, URIRef):
            props.add(str(p))
    for p in graph.subjects(RDF.type, OWL.DatatypeProperty):
        if isinstance(p, URIRef):
            props.add(str(p))
    return props

def extract_property_triples(graph: Graph, properties: Set[str]) -> List[Tuple[str, str, str]]:
    triples = []
    for prop_uri in properties:
        prop = URIRef(prop_uri)
        for domain in graph.objects(prop, RDFS.domain):
            if isinstance(domain, URIRef):
                triples.append((prop_uri, str(RDFS.domain), str(domain)))
        for rng in graph.objects(prop, RDFS.range):
            if isinstance(rng, URIRef):
                triples.append((prop_uri, str(RDFS.range), str(rng)))
        triples.append((prop_uri, str(RDF.type), str(OWL.ObjectProperty) if (prop_uri in properties) else str(OWL.DatatypeProperty)))
    return triples

def extract_relationship_triples(graph: Graph, properties: Set[str]) -> List[Tuple[str, str, str]]:
    triples = []
    for prop_uri in properties:
        prop = URIRef(prop_uri)
        for s, o in graph.subject_objects(prop):
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                triples.append((str(s), prop_uri, str(o)))
    return triples

def generate_ontology_graph_and_triples(ontology_path: str) -> Tuple[Graph, List[Tuple[str, str, str]]]:
    g = Graph()
    g.parse(ontology_path, format='turtle')
    classes = extract_classes(g)
    """ missing_classes = known_classes - classes
    if missing_classes:
        print(f"[Warning] Missing classes in ontology {ontology_path}: {missing_classes}") """
    subclass_triples = extract_subclass_triples(g)
    properties = extract_properties(g)
    domain_range_triples = extract_property_triples(g, properties)
    relationship_triples = extract_relationship_triples(g, properties)
    clean_graph = Graph()
    for cls in classes:
        clean_graph.add((URIRef(cls), RDF.type, OWL.Class))
    for s, p, o in subclass_triples:
        add_triple_safe(clean_graph, s, p, o)
    for s, p, o in domain_range_triples:
        add_triple_safe(clean_graph, s, p, o)
    for s, p, o in relationship_triples:
        add_triple_safe(clean_graph, s, p, o)
    triples = extract_all_triples(clean_graph)
    
    return clean_graph, triples

def generate_knowledge_graph_from_metamodel(source_model_json) -> Tuple[Graph, List[Tuple[str, str, str]]]:
    BASE = Namespace("http://metamodel#")
    g = Graph()
    g.bind("meta", BASE)

    for cls in source_model_json:
        class_uri = BASE[cls["name"]]
        g.add((class_uri, RDF.type, RDFS.Class))
        g.add((class_uri, RDFS.label, Literal(cls["name"])))
        g.add((class_uri, RDFS.comment, Literal(cls["description"])))

        for attr in cls.get("primitive_attributes", []):
            if ":" in attr:
                attr_name, attr_type = map(str.strip, attr.split(":"))
                prop_uri = BASE[f"{cls['name']}_{attr_name}"]
                g.add((prop_uri, RDF.type, RDF.Property))
                g.add((prop_uri, RDFS.domain, class_uri))
                g.add((prop_uri, RDFS.range, Literal(attr_type)))

        for ref in cls.get("reference_attributes", []):
            if ":" in ref:
                ref_name, ref_type_raw = map(str.strip, ref.split(":"))
                ref_type = ref_type_raw.replace("[]", "")
                prop_uri = BASE[f"{cls['name']}_{ref_name}"]
                target_uri = BASE[ref_type]
                g.add((prop_uri, RDF.type, RDF.Property))
                g.add((prop_uri, RDFS.domain, class_uri))
                g.add((prop_uri, RDFS.range, target_uri))

    triples = [(str(s), str(p), str(o)) for s, p, o in g]
    return g, triples

# ------------------ EXTRACT CONTEXT ------------------


def enrich_context_with_graph_walk(graph: Graph, class_uri: str, hops: int = 2) -> List[str]:
    """
    Walk the RDF graph to collect context from related nodes and their labels.
    """
    context = set()
    frontier = {URIRef(class_uri)}
    seen = set()

    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            if node in seen:
                continue
            seen.add(node)

            for _, predicate, obj in graph.triples((node, None, None)):
                if isinstance(obj, URIRef):
                    # Get label for target node
                    label = get_best_label(graph, obj)
                    context.add(f"{extract_local_name(str(predicate))}: {label}")
                    next_frontier.add(obj)

        frontier = next_frontier

    return list(context)

def extract_class_contexts(graph: Graph, hops: int = 2) -> Dict[str, str]:
    """
    Generate a context dictionary for each class using labels and local graph structure.
    """
    class_contexts = {}
    for s in graph.subjects(RDF.type, OWL.Class):
        if isinstance(s, BNode):
            continue

        uri_str = str(s)
        label = get_best_label(graph, s)
        related_context = enrich_context_with_graph_walk(graph, uri_str, hops=hops)

        combined_context = [f"label: {label}"] + related_context
        class_contexts[uri_str] = "\n".join(combined_context)

    return class_contexts

def extract_class_contexts_metamodel(graph: Graph) -> Dict[str, str]:
    class_contexts = {}
    for s in graph.subjects(RDF.type, RDFS.Class):  # ← FIXED HERE
        if isinstance(s, BNode):
            continue
        label = graph.value(subject=s, predicate=RDFS.label)
        label_text = str(label) if label else str(s).split("#")[-1]
        extra_context = enrich_context_with_graph_walk(graph, str(s), hops=HOPS_FOR_CONTEXT)
        context = [label_text] + extra_context
        class_contexts[str(s)] = "\n".join(context)
    return class_contexts

#  ------------------ EMBEDDINGS & TOP-K ------------------


def extract_local_name(uri: str) -> str:
    """
    Extract local name from URI (after last '#' or '/').
    """
    if '#' in uri:
        return uri.rsplit('#', 1)[-1]
    else:
        return uri.rstrip('/').rsplit('/', 1)[-1]

def extract_classes(graph: Graph) -> Set[str]:
    """
    Return a set of all class URIs (as strings) in the graph,
    i.e. everything typed rdf:type owl:Class or rdf:type rdfs:Class.
    """
    classes = set()
    for c in graph.subjects(RDF.type, OWL.Class):
        if isinstance(c, URIRef):
            classes.add(str(c))
    for c in graph.subjects(RDF.type, RDFS.Class):
        if isinstance(c, URIRef):
            classes.add(str(c))
    return classes

def get_label_or_local(graph: Graph, cls_ref: URIRef) -> str:
    """
    Prefer rdfs:label if present; otherwise fallback to local name.
    Underscores → spaces, lowercased.
    """
    label_node = graph.value(cls_ref, RDFS.label)
    if isinstance(label_node, Literal):
        text = str(label_node)
    else:
        text = extract_local_name(str(cls_ref)).replace('_', ' ')
    return text.lower().strip()

def get_best_label(graph: Graph, cls_ref: URIRef) -> str:
    """
    Try to extract meaningful label from common predicates; fallback to local name.
    """
    label_predicates = [
        RDFS.label,
        URIRef("http://www.w3.org/2004/02/skos/core#prefLabel"),
        URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym")
    ]
    for pred in label_predicates:
        label = graph.value(cls_ref, pred)
        if isinstance(label, Literal):
            return str(label).strip().lower()

    # Fallback: local name
    return extract_local_name(str(cls_ref)).replace("_", " ").replace("-", " ").lower()

def extract_tokens_from_uri(uri: str) -> str:
    local = extract_local_name(uri)
    spaced = re.sub(r'[_\-]', ' ', local)
    return split_camel_case(spaced).lower()

def build_rich_text_and_props(graph: Graph, uri: str) -> Tuple[str, Set[str]]:
    """
    Enriched fallback for sparse ontologies. If no labels/comments, uses URI tokenization.
    """
    cls_ref = URIRef(uri)

    label_text = get_best_label(graph, cls_ref)

    comment_node = graph.value(cls_ref, RDFS.comment)
    comment_text = str(comment_node).strip().lower() if isinstance(comment_node, Literal) else ""

    # Superclasses
    supers = []
    for sup in graph.objects(cls_ref, RDFS.subClassOf):
        if isinstance(sup, URIRef):
            supers.append(get_best_label(graph, sup))
    super_text = " subclasses: " + ", ".join(supers) if supers else ""

    # Properties
    prop_texts = []
    prop_set = set()
    for prop in graph.subjects(RDFS.domain, cls_ref):
        prop_label = get_best_label(graph, prop)
        prop_set.add(prop_label)
        range_val = graph.value(prop, RDFS.range)
        if isinstance(range_val, URIRef):
            range_name = get_best_label(graph, range_val)
            prop_texts.append(f"{prop_label}: {range_name}")
        else:
            prop_texts.append(prop_label)
    attrs_text = ". attributes: " + ", ".join(prop_texts) if prop_texts else ""

    # Fallback: if everything is still sparse
    if not label_text or label_text.isnumeric():
        label_text = extract_tokens_from_uri(uri)
    if not comment_text and not super_text and not attrs_text:
        comment_text = f"related to {label_text}"

    rich_doc = " ".join([label_text, comment_text, super_text, attrs_text]).strip()
    return rich_doc, prop_set


def generate_embeddings_and_props(
    graph: Graph,
    entities: Set[str],
    model_name: str = 'BAAI/bge-large-en-v1.5'
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Set[str]]]:
    """
    For each class URI:
      - label_embs: embedding of human-readable label (from rdfs:label if available)
      - rich_embs: embedding of enriched context including label, comments, superclasses, and properties
      - prop_sets: set of property labels

    Returns three dictionaries.
    """
    model = SentenceTransformer(model_name)
    classes_in_graph = extract_classes(graph)

    uris_list: List[str] = []
    label_texts: List[str] = []
    rich_texts: List[str] = []
    prop_sets: Dict[str, Set[str]] = {}

    for uri in entities:
        uris_list.append(uri)
        cls_ref = URIRef(uri)

        # ----------------- 1. Label Text -----------------
        label_node = graph.value(cls_ref, RDFS.label)
        if isinstance(label_node, Literal):
            label_only = str(label_node).strip().lower()
        else:
            label_only = extract_local_name(uri).replace("_", " ").replace("-", " ").lower()

        label_texts.append(f"{cls_ref}: {label_only}")

        # ----------------- 2. Rich Text -----------------
        if uri in classes_in_graph:
            rich_doc, props = build_rich_text_and_props(graph, uri)
            # Always prepend human-readable label at the beginning
            rich_texts.append(f"{cls_ref}:{label_only}: {rich_doc}")
            prop_sets[uri] = props
        else:
            # Non-class fallback
            rich_texts.append(label_only)
            prop_sets[uri] = set()
    #print("label Text:",label_texts)
    #print("rich Text:",rich_texts)
    # ----------------- 3. Embedding -----------------
    label_embeddings = model.encode(label_texts, normalize_embeddings=True)
    rich_embeddings  = model.encode(rich_texts, normalize_embeddings=True)

    # ----------------- 4. Return Dicts -----------------
    label_embs = {uri: emb for uri, emb in zip(uris_list, label_embeddings)}
    rich_embs  = {uri: emb for uri, emb in zip(uris_list, rich_embeddings)}

    return label_embs, rich_embs, prop_sets


def combined_similarity(
    uri1: str,
    uri2: str,
    label_embs1: Dict[str, np.ndarray],
    rich_embs1: Dict[str, np.ndarray],
    props1: Dict[str, set[str]],
    label_embs2: Dict[str, np.ndarray],
    rich_embs2: Dict[str, np.ndarray],
    props2: Dict[str, set[str]],
    alpha: float = 0.5,
    beta: float = 0.3,
    jw_threshold: float = 0.9
) -> float:
    """
    Compute final similarity between two URIs via:
      1. emb_label1, emb_rich1  = embeddings for uri1
         emb_label2, emb_rich2  = embeddings for uri2
      2. combined_embed_score = cos( alpha·emb_label + (1-alpha)·emb_rich , same for uri2 )
      3. lexical Jaro-Winkler on their local names: if > jw_threshold, bump to 0.95
      4. graph Jaccard over property sets:
          jaccard = |props1 ∩ props2| / |props1 ∪ props2|  (0 if both empty)
      5. final = beta·combined_embed_score + (1 - beta)·jaccard
    """
    # 1. Combine embeddings
    emb1 = alpha * label_embs1[uri1] + (1 - alpha) * rich_embs1[uri1]
    emb2 = alpha * label_embs2[uri2] + (1 - alpha) * rich_embs2[uri2]
    emb_score = 1 - cosine(emb1, emb2)  # cosine similarity

    # 2. Lexical boost if labels nearly identical
    l1 = extract_local_name(uri1).replace('_', ' ').lower()
    l2 = extract_local_name(uri2).replace('_', ' ').lower()
    jw = jellyfish.jaro_winkler_similarity(l1, l2)
    if jw > jw_threshold:
        emb_score = max(emb_score, 0.95)

    # 3. Jaccard on property sets
    p1 = props1.get(uri1, set())
    p2 = props2.get(uri2, set())
    if p1 or p2:
        jaccard = len(p1 & p2) / len(p1 | p2)
    else:
        jaccard = 0.0
    final_score=beta * emb_score + (1 - beta) * jaccard
    textual_score=beta * emb_score
    structural_score=(1 - beta) * jaccard
    # 4. Final combined
    return final_score,textual_score,structural_score

#  ------------------ DRIFT SCORE ------------------

def calculate_metamodel_to_metamodel_drift(
    label_src: Dict[str, np.ndarray],
    rich_src: Dict[str, np.ndarray],
    props_src: Dict[str, set],
    label_tgt: Dict[str, np.ndarray],
    rich_tgt: Dict[str, np.ndarray],
    props_tgt: Dict[str, set],
    alpha: float = 0.5,
    beta: float = 0.3,
    jw_threshold: float = 0.9
) -> Dict[str, float]:
    """
    For each class URI in the source metamodel, compute drift score
    with respect to the most similar class in the target ontology.
    
    Drift = 1 - max(similarity)
    """

    drift_scores: Dict[str, float] = {}

    def localname(uri: str) -> str:
        return uri.rsplit('#', 1)[-1].rsplit('/', 1)[-1].replace('_', ' ').lower()

    for src_uri in label_src:
        best_sim = 0.0
        print("Source URI:", src_uri)
        for tgt_uri in label_tgt:
            # Mixed embeddings
            emb1 = alpha * label_src[src_uri] + (1 - alpha) * rich_src[src_uri]
            emb2 = alpha * label_tgt[tgt_uri] + (1 - alpha) * rich_tgt[tgt_uri]

            # Cosine similarity
            emb_sim = 1.0 - cosine(emb1, emb2)

            # Jaro-Winkler lexical boost
            jw = jellyfish.jaro_winkler_similarity(localname(src_uri), localname(tgt_uri))
            if jw > jw_threshold:
                emb_sim = max(emb_sim, 0.95)

            # Jaccard on properties
            p1 = props_src.get(src_uri, set())
            p2 = props_tgt.get(tgt_uri, set())
            jacc = len(p1 & p2) / len(p1 | p2) if p1 or p2 else 0.0

            # Combined similarity
            sim = beta * emb_sim + (1.0 - beta) * jacc
            
            print("\n Textual Features", beta * emb_sim)
            print("\n Structural Features", (1.0 - beta) * jacc)

            if sim > best_sim:
                best_sim = sim

        drift_scores[src_uri] = 1 - best_sim

    return drift_scores


def calculate_drift_with_feature_attribution(
    label_src: Dict[str, np.ndarray],
    rich_src: Dict[str, np.ndarray],
    props_src: Dict[str, set],
    label_tgt: Dict[str, np.ndarray],
    rich_tgt: Dict[str, np.ndarray],
    props_tgt: Dict[str, set],
    alpha: float = 0.5,
    beta: float = 0.3,
    jw_threshold: float = 0.9
) -> Dict[str, Dict[str, float]]:
    """
    For each class in the source, find best match in target and
    decompose the drift into textual vs structural contributions.
    
    Returns:
        {
            class_uri: {
                "drift": float,
                "textual_drift": float,
                "structural_drift": float,
                "best_match": str,
                "textual_similarity": float,
                "structural_similarity": float
            }
        }
    """

    def localname(uri: str) -> str:
        return uri.rsplit('#', 1)[-1].rsplit('/', 1)[-1].replace('_', ' ').lower()

    drift_report: Dict[str, Dict[str, float]] = {}

    for src_uri in label_src:
        best_sim = 0.0
        best_match = None
        best_text_sim = 0.0
        best_struct_sim = 0.0

        for tgt_uri in label_tgt:
            emb1 = alpha * label_src[src_uri] + (1 - alpha) * rich_src[src_uri]
            emb2 = alpha * label_tgt[tgt_uri] + (1 - alpha) * rich_tgt[tgt_uri]

            emb_sim = 1.0 - cosine(emb1, emb2)

            jw = jellyfish.jaro_winkler_similarity(localname(src_uri), localname(tgt_uri))
            if jw > jw_threshold:
                emb_sim = max(emb_sim, 0.95)

            p1 = props_src.get(src_uri, set())
            p2 = props_tgt.get(tgt_uri, set())
            jacc = len(p1 & p2) / len(p1 | p2) if p1 or p2 else 0.0

            sim = beta * emb_sim + (1.0 - beta) * jacc

            if sim > best_sim:
                best_sim = sim
                best_match = tgt_uri
                best_text_sim = emb_sim
                best_struct_sim = jacc

        drift_report[src_uri] = {
            "drift": 1.0 - best_sim,
            "best_match": best_match,
            "textual_drift": 1.0 - best_text_sim,
            "structural_drift": 1.0 - best_struct_sim,
            "overall_similarity": best_sim,
            "textual_similarity": best_text_sim,
            "structural_similarity": best_struct_sim
        }

    return drift_report



def detect_metamodel_drift_with_ontology(
    mm1_graph: Graph,
    mm2_graph: Graph,
    ontology_graph: Graph,
    alpha: float = 0.5,
    beta: float = 0.3,
    jw_threshold: float = 0.8
) -> Dict[str, float]:
    """
    Compare MM1 and MM2 similarity to ontology and return drift scores per class.
    Drift is the average difference in similarity across all ontology classes for aligned classes.
    """
    # Extract class URIs
    mm1_classes = extract_classes(mm1_graph)
    mm2_classes = extract_classes(mm2_graph)
    ont_classes = extract_classes(ontology_graph)

    # Generate embeddings
    label1, rich1, props1 = generate_embeddings_and_props(mm1_graph, mm1_classes)
    label2, rich2, props2 = generate_embeddings_and_props(mm2_graph, mm2_classes)
    label_o, rich_o, props_o = generate_embeddings_and_props(ontology_graph, ont_classes)

    # Match MM1 and MM2 classes by name using relaxed matching
    aligned_classes = {}
    for uri1 in mm1_classes:
        name1 = extract_local_name(uri1).lower().replace('_', ' ')
        matched_uri2 = next(
            (uri2 for uri2 in mm2_classes
             if extract_local_name(uri2).lower().replace('_', ' ') == name1),
            None
        )

        if not matched_uri2:
            best_score = 0.0
            for uri2 in mm2_classes:
                name2 = extract_local_name(uri2).lower().replace('_', ' ')
                score = jellyfish.jaro_winkler_similarity(name1, name2)
                if score > best_score:
                    best_score = score
                    matched_uri2 = uri2 if score > jw_threshold else None

        if matched_uri2:
            aligned_classes[name1] = (uri1, matched_uri2)

    # Compute drift as average difference across all ontology class similarities
    drift_combined = {}
    drift_text = {}
    drift_struct = {}
    for name, (uri1, uri2) in aligned_classes.items():
        diffs_combined = []
        diffs_text = []
        diffs_struct = []
        for ont_uri in ont_classes:
            sim1, text1, struct1 = combined_similarity(uri1, ont_uri, label1, rich1, props1, label_o, rich_o, props_o,
                                       alpha=alpha, beta=beta, jw_threshold=jw_threshold)
            sim2, text2, struct2 = combined_similarity(uri2, ont_uri, label2, rich2, props2, label_o, rich_o, props_o,
                                       alpha=alpha, beta=beta, jw_threshold=jw_threshold)
            diffs_combined.append(abs(sim2 - sim1))
            diffs_text.append(abs(text2 - text1))
            diffs_struct.append(abs(struct2 - struct1))

        drift_score_combined = sum(diffs_combined) / len( diffs_combined) if  diffs_combined else 0.0
        drift_combined[name] = drift_score_combined
        
        drift_score_text = sum(diffs_text) / len(diffs_text) if diffs_text else 0.0
        drift_text[name] = drift_score_text
        
        drift_score_struct = sum(diffs_struct) / len(diffs_struct) if diffs_struct else 0.0
        drift_struct[name] = drift_score_struct

    return drift_combined, drift_text, drift_struct






#  ------------------ MAIN ------------------

def main():
    print("\n1. Loading metamodel version 1 ...")
    with open('C:\\Users\\abbasi\\Desktop\\drift-mgt-mddt\\output\\metamodel-v1.json', 'r', encoding='utf-8') as f:
        MM_v1 = json.load(f)
    MM1_graph, MM1_ttl = generate_knowledge_graph_from_metamodel(MM_v1)
    print(f" Metamodel 1 triples count: {len(MM1_ttl)}")
    
    print("\n2. Loading metamodel version 2 ...")
    with  open('C:\\Users\\abbasi\\Desktop\\drift-mgt-mddt\\output\\metamodel-v2.json', 'r', encoding='utf-8') as f:
        MM_v2 = json.load(f)
    MM2_graph, MM2_ttl = generate_knowledge_graph_from_metamodel(MM_v2)
    print(f" Metamodel 2 triples count: {len(MM2_ttl)}")
        
    print("\n3. Processing ontology...")
    target_onto_graph, target_onto_ttl = generate_ontology_graph_and_triples(TARGET_OWL_PATH)
    print(f" Ontology triples count: {len(target_onto_ttl)}")
    
    MM1_classes = extract_classes(MM1_graph)
    MM2_classes = extract_classes(MM2_graph)
    onto_classes = extract_classes(target_onto_graph)
    
     
    label_mm1, rich_mm1, props_mm1 = generate_embeddings_and_props(MM1_graph, MM1_classes)
    label_mm2, rich_mm2, props_mm2 = generate_embeddings_and_props(MM2_graph, MM2_classes)
    
    
    # drift_mm1_to_onto = calculate_metamodel_to_metamodel_drift(
    #     label_src=label_mm1,
    #     rich_src=rich_mm1,
    #     props_src=props_mm1,
    #     label_tgt=label_mm2,
    #     rich_tgt=rich_mm2,
    #     props_tgt=props_mm2,
    #     alpha=0.5,
    #     beta=0.3,
    #     jw_threshold=0.9
    # )

    # print("\n--- Drift between Metamodels ---\n")
    # for uri, drift in sorted(drift_mm1_to_onto.items(), key=lambda x: -x[1]):
    #     print(f"{uri} -> drift: {drift:.6f}\n")
    
    print("\n4. Drift Summary Comparing Two Metamodels (MM1 and MM2)---\n")
    drift_report = calculate_drift_with_feature_attribution(
         label_src=label_mm1,
         rich_src=rich_mm1,
         props_src=props_mm1,
         label_tgt=label_mm2,
         rich_tgt=rich_mm2,
         props_tgt=props_mm2,
         alpha=0.5,
         beta=0.3,
         jw_threshold=0.9
    )
    
    for uri, drift_info in drift_report.items():
        print(f"Class: {uri}")
        print(f"  Best match: {drift_info['best_match']}")
        print(f"  Drift Score: {drift_info['drift']:.3f}")
        print(f"    - Textual Drift: {drift_info['textual_drift']:.3f}")
        print(f"    - Structural Drift: {drift_info['structural_drift']:.3f}")
        print(f"  Overall Similarity: {drift_info['overall_similarity']:.3f}")
        print(f"    - Textual Similarity: {drift_info['textual_similarity']:.3f}")
        print(f"    - Structural Similarity: {drift_info['structural_similarity']:.3f}")
        print()
    
    
    
    print("\n5. Computing Drift between MM1 and MM2 w.r.t Ontology ...")
    drift_combined, drift_text, drift_struct  = detect_metamodel_drift_with_ontology(MM1_graph, MM2_graph, target_onto_graph)

    print("\n--- Drift Summary Collective Score ---\n")
    for cls, drift in sorted(drift_combined.items(), key=lambda x: -x[1]):
        print(f"{cls}: drift = {drift:.6f}\n") 
        
    #print("\n--- Textual Drift Summary ---\n")
    #for cls, drift in sorted(drift_text.items(), key=lambda x: -x[1]):
    #    print(f"{cls}: drift = {drift:.6f}\n") 
    
    #print("\n--- Structural Drift Summary ---\n") 
    #for cls, drift in sorted(drift_struct.items(), key=lambda x: -x[1]):
    #    print(f"{cls}: drift = {drift:.6f}\n") 
     
   
main()