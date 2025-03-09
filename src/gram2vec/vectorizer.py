from collections import Counter
import demoji
import time
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Callable, Optional, Iterable

from ._load_spacy import nlp, Doc
from ._load_vocab import vocab

def get_feature_counts(doc):
    feature_types = ["pos_tags", 
                     "pos_verbs",
                     "pos_adjectives",
                     "pos_adverbs",
                     "pos_proper_nouns",
                     "pos_adpositions",
                     "dep_labels", 
                     "morph_tags", 
                     "pos_bigrams", 
                     "sentences", 
                     "func_words", 
                     "punctuation",
                     "punct_periods",
                     "punct_commas",
                     "punct_colons",
                     "punct_semicolons",
                     "punct_exclamations",
                     "punct_questions",
                     "letters", 
                     "tokens", 
                     "named_entities",
                     "NEs_person",
                     "NEs_location_loc",
                     "NEs_location_gpe",
                     "NEs_organization",
                     "NEs_date",
                     "NEs_without_date",
                     "token_VB",
                     "token_VBD",
                     "token_VBG",
                     "token_VBN",
                     "token_VBP",
                     "token_VBZ",
                     "token_EX",
                     "token_FW",
                     "token_PRP",
                     "token_superlatives",
                     "token_comparatives",
                     "first_second_person_pronouns",
                     "third_person_pronouns",
                     "pronoun_it"
                     ]
    feature_counts = {}
    
    for feature in feature_types:
        feature_list = getattr(doc._, feature)
        feature_counts[feature] = len(feature_list)
    return feature_counts

def measure_time(func):
    """Debugging function for measuring function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds.")
        return result
    return wrapper

#~~~ Features ~~~

@dataclass
class Document:
    """
    Encapsulates the raw text and spacy doc. Needed because emojis must be taken out of the spacy doc before 
    the dependency parse, but the emojis feature still needs access to the emojis from the text
    """
    raw:str
    doc:Doc
    num_tokens:int
    
REGISTERD_FEATURES = {}

class Feature:
    """Encapsulates a feature counting function. When the function is called, normalization is applied to the counted features"""
    def __init__(self, func:Callable):
        self.func = func    
        self.name = func.__name__
        
    def __call__(self, doc, vocab=None):
        counted_features = self.func(doc)
        if vocab is not None:
            all_counts = self._include_zero_vocab_counts(counted_features, vocab)
        else:
            all_counts = pd.Series(counted_features)
        normalized_counts = self._normalize(all_counts, doc.num_tokens)
        return self._prefix_feature_names(normalized_counts)
    
    @classmethod
    def register(cls, func):
        """Creates a Feature object and registers it to the REGISTERED_FEATURES dict"""
        func = cls(func)
        REGISTERD_FEATURES[func.name] = func
        return func

    def _include_zero_vocab_counts(self, counted_features:Counter, vocab:Tuple[str]) -> pd.Series:
        """Includes the vocabulary items that were not counted in the document (to ensure the same size vector for all documents)"""
        count_dict = {}
        for feature in vocab:
            if feature in counted_features:
                count = counted_features[feature] 
            else:
                count = 0
            count_dict[feature] = count
        return pd.Series(count_dict)
    
    def _get_sum(self, counts:pd.Series) -> int:
        """Gets sum of counts. Accounts for possible zero counts"""
        return sum(counts) if sum(counts) > 0 else 1

    def _normalize(self, counts:pd.Series, num_tokens:int) -> pd.Series:
        """Normalizes each count by the sum of counts for that feature"""
        if self.name in ["num_tokens", "avg_chars_per_token", "avg_tokens_per_sentence", "avg_noun_chunk_length", "avg_verb_chunk_length"]:
            return counts  
        if self.name in ["emojis", 
                         "punctuation", 
                         "punct_periods",
                         "punct_commas",
                         "punct_colons",
                         "punct_semicolons",
                         "punct_exclamations",
                         "punct_questions",
                         "func_words", 
                         "sentences", 
                         "named_entities", 
                         "NEs_person", 
                         "NEs_location_loc", 
                         "NEs_location_gpe", 
                         "NEs_organization", 
                         "NEs_date", 
                         "NEs_except_date",
                         "pos_verbs",
                         "pos_adjectives",
                         "pos_adverbs",
                         "pos_proper_nouns",
                         "pos_adpositions",
                         "token_VB",
                         "token_VBD",
                         "token_VBG",
                         "token_VBN",
                         "token_VBP",
                         "token_VBZ",
                         "token_EX",
                         "token_FW",
                         "token_PRP",
                         "token_superlatives",
                         "token_comparatives",
                         "first_second_person_pronouns",
                         "third_person_pronouns",
                         "pronoun_it"
                         ]:
            return counts / num_tokens
        else:
            return counts / self._get_sum(counts)
    
    def _prefix_feature_names(self, features:pd.Series) -> pd.Series:
        """
        For each low level feature, prefix the name of the high level feature to it z
                                EXAMPLE:  ADJ -> pos_unigrams:ADJ
        """
        return features.add_prefix(f"{self.name}:")
        
@Feature.register
def pos_unigrams(text:Document) -> Feature:
    return Counter(text.doc._.pos_tags)
    
@Feature.register
def pos_bigrams(text:Document) -> Feature:
    return Counter(text.doc._.pos_bigrams)

@Feature.register
def func_words(text:Document) -> Feature:
    return Counter(text.doc._.func_words)

@Feature.register
def pos_verbs(text:Document) -> Feature:
    return Counter(text.doc._.pos_verbs)

@Feature.register
def pos_adjectives(text:Document) -> Feature:
    return Counter(text.doc._.pos_adjectives)

@Feature.register
def pos_adverbs(text:Document) -> Feature:
    return Counter(text.doc._.pos_adverbs)

@Feature.register
def pos_proper_nouns(text:Document) -> Feature:
    return Counter(text.doc._.pos_proper_nouns)

@Feature.register
def pos_adpositions(text:Document) -> Feature:
    return Counter(text.doc._.pos_adpositions)

@Feature.register
def punctuation(text:Document) -> Feature:
    return Counter(text.doc._.punctuation)

@Feature.register   
def punct_periods(text:Document) -> Feature:
    return Counter(text.doc._.punct_periods)

@Feature.register
def punct_commas(text:Document) -> Feature:
    return Counter(text.doc._.punct_commas)

@Feature.register
def punct_colons(text:Document) -> Feature:
    return Counter(text.doc._.punct_colons)

@Feature.register
def punct_semicolons(text:Document) -> Feature:
    return Counter(text.doc._.punct_semicolons)

@Feature.register
def punct_exclamations(text:Document) -> Feature:
    return Counter(text.doc._.punct_exclamations)

@Feature.register
def punct_questions(text:Document) -> Feature:
    return Counter(text.doc._.punct_questions)

@Feature.register
def letters(text:Document) -> Feature:
    return Counter(text.doc._.letters)

@Feature.register
def dep_labels(text:Document) -> Feature:
    return Counter(text.doc._.dep_labels)

@Feature.register
def morph_tags(text:Document) -> Feature:
    return Counter(text.doc._.morph_tags)

@Feature.register
def sentences(text:Document) -> Feature:
    return Counter(text.doc._.sentences)

# emojis must get removed before processed through spaCy,
# so spaCy extensions cannot be used here unfortunately
@Feature.register
def emojis(text:Document) -> Feature:
    emojis_vocab = vocab.get("emojis")
    extracted_emojis = demoji.findall_list(text.raw, desc=False)
    counted_emojis = Counter()

    for emoji in extracted_emojis:
        if emoji in emojis_vocab:
            counted_emojis[emoji] += 1
        else:
            counted_emojis["OOV_emoji"] += 1
    return counted_emojis

@Feature.register
def num_tokens(text:Document) -> Feature:
    return Counter({"num_tokens": text.num_tokens})

@Feature.register
def named_entities(text:Document) -> Feature:
    return Counter(text.doc._.named_entities)

@Feature.register
def NEs_person(text:Document) -> Feature:
    return Counter(text.doc._.NEs_person)

@Feature.register
def NEs_location_loc(text:Document) -> Feature:
    return Counter(text.doc._.NEs_location_loc)

@Feature.register
def NEs_location_gpe(text:Document) -> Feature:
    return Counter(text.doc._.NEs_location_gpe)

@Feature.register
def NEs_organization(text:Document) -> Feature:
    return Counter(text.doc._.NEs_organization)

@Feature.register
def NEs_date(text:Document) -> Feature:
    return Counter(text.doc._.NEs_date)

@Feature.register
def NEs_except_date(text:Document) -> Feature:
    return Counter(text.doc._.NEs_except_date)

@Feature.register
def token_VB(text:Document) -> Feature:
    return Counter(text.doc._.token_VB)

@Feature.register
def token_VBD(text:Document) -> Feature:
    return Counter(text.doc._.token_VBD)

@Feature.register
def token_VBG(text:Document) -> Feature:
    return Counter(text.doc._.token_VBG)

@Feature.register
def token_VBN(text:Document) -> Feature:
    return Counter(text.doc._.token_VBN)

@Feature.register
def token_VBP(text:Document) -> Feature:
    return Counter(text.doc._.token_VBP)

@Feature.register
def token_VBZ(text:Document) -> Feature:
    return Counter(text.doc._.token_VBZ)

@Feature.register
def token_EX(text:Document) -> Feature:
    return Counter(text.doc._.token_EX)

@Feature.register
def token_FW(text:Document) -> Feature:
    return Counter(text.doc._.token_FW)

@Feature.register
def token_PRP(text:Document) -> Feature:
    return Counter(text.doc._.token_PRP)

@Feature.register
def token_superlatives(text:Document) -> Feature:
    superlatives = Counter()
    for token in text.doc:
        if token.tag_ in ["JJS", "RBS"]:
            superlatives[token.text] += 1
    return superlatives

@Feature.register
def token_comparatives(text:Document) -> Feature:
    comparatives = Counter()
    for token in text.doc:
        if token.tag_ in ["JJR", "RBR"]:
            comparatives[token.text] += 1
    return comparatives

@Feature.register
def pronoun_person(text: Document) -> Feature:
    pronoun_counter = Counter()
    for token in text.doc:
        if token.pos_ == "PRON":  # Check if the token is a pronoun
            person = token.morph.get("Person")
            if person:
                person_label = f"person_{person[0]}"
                pronoun_counter[person_label] += 1
    
    return pronoun_counter

@Feature.register
def first_second_person_pronouns(text: Document) -> Feature:
    pronoun_counter = Counter()
    for token in text.doc:
        if token.pos_ == "PRON":
            person = token.morph.get("Person")
            if person and person[0] in {"1", "2"}:
                pronoun_counter["1st_2nd_person"] += 1
    return pronoun_counter

@Feature.register
def third_person_pronouns(text: Document) -> Feature:
    """
    Counts 3rd person pronouns, excluding 'it'
    """
    pronoun_counter = Counter()
    for token in text.doc:
        if token.pos_ == "PRON":
            person = token.morph.get("Person")
            if person and person[0] == "3" and token.text.lower() != "it":
                pronoun_counter["3rd_person"] += 1
    return pronoun_counter

@Feature.register
def pronoun_it(text: Document) -> Feature:
    pronoun_counter = Counter()
    for token in text.doc:
        if token.pos_ == "PRON" and token.text.lower() == "it":
            pronoun_counter["pronoun_it"] += 1
    return pronoun_counter

@Feature.register
def avg_chars_per_token(text: Document) -> Feature:
    total_characters = sum(len(token.text) for token in text.doc)
    num_tokens =text.num_tokens
    avg_chars_per_token = total_characters / num_tokens if num_tokens > 0 else 0
    return Counter({"avg_chars_per_token": avg_chars_per_token})

@Feature.register
def avg_tokens_per_sentence(text: Document) -> Feature:
    total_tokens = text.num_tokens
    num_sentences = len(list(text.doc.sents))
    avg_tokens_per_sentence = total_tokens / num_sentences if num_sentences > 0 else 0
    return Counter({"avg_tokens_per_sentence": avg_tokens_per_sentence})

@Feature.register
def avg_noun_chunk_length(text: Document) -> Feature:
    noun_chunks = list(text.doc.noun_chunks)
    total_chunk_length = sum(len(chunk.text) for chunk in noun_chunks)
    num_chunks = len(noun_chunks)
    avg_chunk_length = total_chunk_length / num_chunks if num_chunks > 0 else 0
    return Counter({"avg_noun_chunk_length": avg_chunk_length})

@Feature.register
def avg_verb_chunk_length(text: Document) -> Feature: #it's no built-in function to retrieve VP in spaCy, so we have do it manually
    verb_chunks = []
    for token in text.doc:
        if token.pos_ == "VERB":
            chunk = [token] + list(token.children)
            chunk = [t for t in chunk if t.dep_ in {"aux", "xcomp", "ccomp", "advcl", "prep", "pobj", "dobj", "nsubj", "nsubjpass", "csubj", "csubjpass", "attr", "acomp"}]
            verb_chunks.append(chunk)

    total_chunk_length = sum(len(chunk) for chunk in verb_chunks)
    num_chunks = len(verb_chunks)
    avg_chunk_length = total_chunk_length / num_chunks if num_chunks > 0 else 0
    return Counter({"avg_verb_chunk_length": avg_chunk_length})
    

# ~~~ Processing ~~~    
def get_activated_features(config:Optional[Dict]) -> List[Feature]:
    """Retrieves activated features from register according to a given config. Falls back to default config if none is provided"""
    if config is None:
        default_config = {
            "pos_unigrams":1,
            "pos_bigrams":1,
            "func_words":1,
            "pos_verbs":1,
            "pos_adjectives":1,
            "pos_adverbs":1,
            "pos_proper_nouns":1,
            "pos_adpositions":1,
            "punctuation":1,
            "punct_periods":1,
            "punct_commas":1,
            "punct_colons":1,
            "punct_semicolons":1,
            "punct_exclamations":1,
            "punct_questions":1,
            "letters":0,
            "emojis":1,
            "dep_labels":1,
            "morph_tags":1,
            "sentences":1,
            "num_tokens":1,
            "named_entities":1,
            "NEs_person":1,
            "NEs_location_loc":1,
            "NEs_location_gpe":1,
            "NEs_organization":1,
            "NEs_date":1,
            "NEs_except_date":1,
            "token_VB":1,
            "token_VBD":1,
            "token_VBG":1,
            "token_VBN":1,
            "token_VBP":1,
            "token_VBZ":1,
            "token_EX":1,
            "token_FW":1,
            "token_PRP":1,
            "token_superlatives":1,
            "token_comparatives":1,
            "first_second_person_pronouns":1,
            "third_person_pronouns":1,
            "pronoun_it":1,
            "avg_chars_per_token":1,
            "avg_tokens_per_sentence":1,
            "avg_noun_chunk_length":1,
            "avg_verb_chunk_length":1
            }
        config = default_config
    return [REGISTERD_FEATURES[feat_name] for feat_name, num in config.items() if num == 1]

def load_jsonlines(path:str) -> pd.DataFrame:
    """Loads 1 or more .jsonl files into a dataframe"""
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    else:
        dfs = [pd.read_json(file, lines=True) for file in Path(path).glob("*.jsonl")]
        return pd.concat(dfs).reset_index(drop=True)
    
def _remove_emojis(document:str) -> str:
    """Removes emojis from a string and fixes spacing issue caused by emoji removal"""
    new_string = demoji.replace(document, "").split()
    return " ".join(new_string)

def _process_documents(documents:Iterable[str]) -> List[Document]:
    """Converts all provided documents into Document instances, which encapsulates the raw text and spacy doc"""
    nlp_docs = nlp.pipe([_remove_emojis(doc) for doc in documents])
    original_token_counts = [len(nlp(doc)) for doc in documents]
    
    processed = []
    for raw_text, nlp_doc, original_token_count in zip(documents, nlp_docs, original_token_counts):
        processed.append(Document(raw_text, nlp_doc, original_token_count))
    return processed

def _get_json_entries(df) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Retrieves the 'fullText', 'authorIDs', and 'documentID' fields from a json-loaded dataframe"""
    try:
        documents = df["fullText"]
        author_ids = df["authorIDs"]
        document_ids = df["documentID"]
    except KeyError:
        raise KeyError("Specified jsonline(s) file missing one or more fields: 'fullText', 'authorIDs', 'documentID'")
    
    return documents, author_ids, document_ids

def _content_embedding(doc:Document) -> pd.Series:
    """Retrieves the spacy document embedding and returns it as a Series object"""
    return pd.Series(doc.doc.vector).add_prefix("Embedding dim: ")
    
def _apply_features(doc: Document, config: Optional[Dict], include_content_embedding: bool) -> pd.Series:
    """Applies all feature extractors to a given document, optionally adding the spaCy embedding vector"""
    features = []
    # List of features that do not require a vocabulary
    no_vocab_features = [
        "num_tokens",
        "avg_chars_per_token",
        "avg_tokens_per_sentence",
        "avg_noun_chunk_length",
        "avg_verb_chunk_length",
        "first_second_person_pronouns",
        "third_person_pronouns",
        "pronoun_it",
        "all_personal_pronouns",
        "token_superlatives",
        "token_comparatives"
    ]
    
    for feature in get_activated_features(config):
        if feature.name in no_vocab_features:
            # Directly call the feature function without vocab
            extraction = feature(doc)
        else:
            feature_vocab = vocab.get(feature.name)
            extraction = feature(doc, feature_vocab)
        features.append(extraction)
    
    if include_content_embedding:
        features.append(_content_embedding(doc))
    
    return pd.concat(features, axis=0)

def _apply_features_to_docs(docs:List[Document],
                            config:Optional[Dict], 
                            include_content_embedding:bool) -> pd.DataFrame:
    """Applies the feature extractors to all documents and creates a style vector matrix"""
    feature_vectors = []
    for doc in docs:
        vector = _apply_features(doc, config, include_content_embedding)
        feature_vectors.append(vector)
    return pd.concat(feature_vectors, axis=1).T

def from_jsonlines(path:str, 
                   config:Optional[Dict]=None, 
                   include_content_embedding=False) -> pd.DataFrame:
    """
    Given a path to either a jsonlines file OR directory of jsonlines files, creates a stylistic feature 
    vector matrix. Document IDs and author IDs are included, retrieved from the provided jsonlines file(s)\n
    Args:
    -----
        path (str): 
            path to a jsonlines file OR directory of jsonlines files
        config(Dict | None): 
            Feature activation configuration. Uses a default if none is provided
        include_content_embedding (bool):
            option to include the word2vec document embedding\n
    Returns:
    -------
        pd.DataFrame: dataframe where rows are documents and columns are low level features
    """
    df = load_jsonlines(path)
    documents, author_ids, document_ids = _get_json_entries(df)
    documents = _process_documents(documents)
    if include_content_embedding:
        print("Gram2Vec: 'include_content_embedding' flag set to True. Including document word2vec embedding...")
        print("Gram2Vec: (WARNING) embedding should only be used for experiments, not attribution")
    vector_df = _apply_features_to_docs(documents, config, include_content_embedding)
    vector_df.insert(0, "authorIDs", author_ids)
    vector_df.set_index(document_ids, inplace=True)
    return vector_df
    
def from_documents(documents:Iterable[str], 
                   config:Optional[Dict]=None, 
                   include_content_embedding=False) -> pd.DataFrame:
    """
    Given an iterable of documents, creates a stylistic feature vector matrix. Document IDs and author IDs are NOT included\n
    Args:
    -----
        documents(Iterable):
            iterable of strings to be converted into a matrix
        config(Dict | None): 
            Feature activation configuration. Uses a default if none is provided
        include_content_embedding (bool):
            option to include the word2vec document embedding\n
    Returns:
    --------
        pd.DataFrame: dataframe where each row is a document and column is a low level feature
    """
    documents = _process_documents(documents)
    if include_content_embedding:
        print("Gram2Vec: 'include_content_embedding' flag set to True. Including document word2vec embedding...")
        print("Gram2Vec: (WARNING) embedding should only be used for experiments, not attribution")
    vector_df = _apply_features_to_docs(documents, config, include_content_embedding)
    return vector_df