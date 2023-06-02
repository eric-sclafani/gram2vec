
from collections import Counter
from dataclasses import dataclass
import demoji
from nltk import bigrams
from nltk import FreqDist
import numpy as np 
import pandas as pd
import os
from typing import Optional, Tuple, List, Dict, Callable
from load_spacy import nlp

# ~~~ Vocab ~~~

def load_txt_file(path:str) -> Tuple[str]:
    with open (path, "r") as fin:
        return tuple(map(lambda x: x.strip("\n"), fin.readlines()))

def vocab_loader(name:str, refresh_non_static_vocab:False) -> Tuple[str]:
    """Given a feature name, loads that feature's vocabulary from disk"""
    
    

            
#~~~ Features ~~~


REGISTERD_FEATURES = {}

class Feature:
    
    def __init__(self, func):
        self.func = func    
        self.name = func.__name__
        
    def __call__(self, doc):
        counted_features = self.func(doc)
        return self.include_zero_vocab_counts(counted_features)

    def include_zero_vocab_counts(self, counted_features:Counter, vocab:Tuple[str]) -> Dict[str, int]:
        count_dict = {}
        for feature in vocab:
            if feature in counted_features:
                count = counted_features[feature] 
            else:
                count = 0
            count_dict[feature] = count
        return count_dict

    @classmethod
    def register(cls, func):
        func = cls(func)
        REGISTERD_FEATURES[func.name] = func
        return func





# @dataclass
# class Feature:
    
#     featurizer_name:str
#     feature_counts:Counter
#     vocab:Tuple[str]
    
#     def counts_to_df(self) -> pd.DataFrame:
#         """Converts a dictionary of counts into a numpy array and normalizes"""
#         counts = list(self.feature_counts.values())
#         return np.array(counts).flatten() / self.normalize_by
    
#     def _is_counted(self, feature:str) -> bool:
#         return feature in self.feature_counts

#     def include_zero_vocab_counts(self, vocab:Tuple[str]) -> Dict[str, int]:
#         """Creates count dict that includes counted items in self._feature_counts and items counted 0 times (to ensure consistent vector lengths)"""
#         count_dict = {}
#         for feature in vocab:
#             if self._is_counted(feature):
#                 count = self._feature_counts[feature] 
#             else:
#                 count = 0
#             count_dict[feature] = count
#         return count_dict

#     def sum_of_counts(counts:Dict) -> int:
#         """Sums the counts of a count dictionary. Returns 1 if counts sum to 0"""
#         count_sum = sum(counts.values())
#         return count_sum if count_sum > 0 else 1


      
@Feature.register
def pos_unigrams(doc) -> Feature:
    
    vocab = load_vocab("vocab/static/pos_unigrams.txt")
    doc_pos_tag_counts = Counter(doc.pos_tags)
    all_pos_tag_counts = add_zero_vocab_counts(vocab, doc_pos_tag_counts)
    
    return Feature("POS Unigram", all_pos_tag_counts, len(doc.pos_tags))

def pos_bigrams(doc) -> Feature:
    
    vocab = load_vocab("vocab/non_static/pan/pos_bigrams/pos_bigrams.pkl", type="non_static")
    doc_pos_bigram_counts = Counter(get_bigrams_with_boundary_syms(doc, doc.pos_tags))
    all_pos_bigram_counts = add_zero_vocab_counts(vocab, doc_pos_bigram_counts)
    
    return Feature("POS Bigram", all_pos_bigram_counts, sum_of_counts(doc_pos_bigram_counts))

def func_words(doc) -> Feature:
    
    vocab = load_vocab("vocab/static/function_words.txt")
    doc_func_word_counts = Counter([token for token in doc.tokens if token in vocab])
    all_func_word_counts = add_zero_vocab_counts(vocab, doc_func_word_counts)
    
    return Feature("Function word", all_func_word_counts, sum_of_counts(doc_func_word_counts))

def punc(doc) -> Feature:
    
    vocab = load_vocab("vocab/static/punc_marks.txt")
    doc_punc_counts = Counter([punc for token in doc.tokens for punc in token if punc in vocab])
    all_punc_counts = add_zero_vocab_counts(vocab, doc_punc_counts)
    
    return Feature("Punctuation", all_punc_counts, sum_of_counts(doc_punc_counts))

def letters(doc) -> Feature:
    
    vocab = load_vocab("vocab/static/letters.txt")
    doc_letter_counts = Counter([letter for token in doc.tokens for letter in token if letter in vocab])
    all_letter_counts = add_zero_vocab_counts(vocab, doc_letter_counts)
    
    return Feature("Letter", all_letter_counts, sum_of_counts(doc_letter_counts))

def common_emojis(doc) -> Feature:
    
    vocab = load_vocab("vocab/static/common_emojis.txt")
    extract_emojis = demoji.findall_list(doc.text, desc=False)
    doc_emoji_counts = Counter(filter(lambda x: x in vocab, extract_emojis))
    all_emoji_counts = add_zero_vocab_counts(vocab, doc_emoji_counts)
    
    return Feature("Emoji", all_emoji_counts, len(doc.tokens))

#! DISABLED FOR NOW (only used for experimentation, NOT feature extraction)
# def embedding_vector(doc) -> Feature:
#     """spaCy word2vec (or glove?) document embedding"""
#     embedding = {"embedding_vector" : doc.spacy_doc.vector}
#     return Feature(None, embedding)

#! doc.words causing crash
def document_stats(doc) -> Feature:
    words = doc.words
    
    #! GETTING SPLIT INTO SEPARATE VECTORS
    #! REPLACE ALL STATISTICAL CALCULATIONS WITH BUILT IN
    
    
    doc_statistics = {"short_words" : len([1 for word in words if len(word) < 5])/len(words) if len(words) else 0, 
                      "large_words" : len([1 for word in words if len(word) > 4])/len(words) if len(words) else 0,
                      "word_len_avg": np.mean([len(word) for word in words]),
                      "word_len_std": np.std([len(word) for word in words]),
                      "sent_len_avg": np.mean([len(sent) for sent in doc.sentences]),
                      "sent_len_std": np.std([len(sent) for sent in doc.sentences]),
                      "hapaxes"     : len(FreqDist(words).hapaxes())/len(words) if len(words) else 0}
    return Feature("Document statistic", doc_statistics)

def dep_labels(doc) -> Feature:

    vocab = load_vocab("vocab/static/dep_labels.txt")
    doc_dep_labels = Counter([dep for dep in doc.dep_labels])
    all_dep_labels = add_zero_vocab_counts(vocab, doc_dep_labels)
    
    return Feature("Dependency label", all_dep_labels, sum_of_counts(doc_dep_labels))

def mixed_bigrams(doc) -> Feature:
    
    vocab = load_vocab("vocab/non_static/pan/mixed_bigrams/mixed_bigrams.pkl", type="non_static")
    doc_mixed_bigrams = Counter(bigrams(replace_openclass(doc.tokens, doc.pos_tags)))
    all_mixed_bigrams = add_zero_vocab_counts(vocab, doc_mixed_bigrams)
    
    return Feature("Mixed Bigram", all_mixed_bigrams, sum_of_counts(doc_mixed_bigrams))

def morph_tags(doc) -> Feature:
    
    vocab = load_vocab("vocab/static/morph_tags.txt")
    doc_morph_tags = Counter(parse_morph([token.morph for token in doc.spacy_doc]))
    all_morph_tags = add_zero_vocab_counts(vocab, doc_morph_tags)
    
    return Feature("Morphology tag", all_morph_tags, sum_of_counts(doc_morph_tags))

# ~~~ Processing ~~~


#! need a refresh parameter

     
class GrammarVectorizer:
    
    
    register = (pos_unigrams,
                pos_bigrams,
                func_words,
                punc,
                letters,
                common_emojis,
                embedding_vector,
                document_stats,
                dep_labels,
                mixed_bigrams,
                morph_tags)
    
    def __init__(self, config:Dict[str,int]=None):
        
        
        self._config = self._process_config(config)
        os.system("./clear_logs.sh")
        
    def get_config(self) -> List[str]:
        """Retrieves the names of all activated features"""
        return [feat.__name__ for feat in self._config]
        
    def _process_config(self, passed_config: Optional[Dict]) -> List:
        """Reads which features to activate and returns a list of featurizer functions"""
        current_config = DEFAULT_CONFIG if not passed_config else passed_config
        activated_feats = []
        for feat in self.register:
            try:
                if current_config[feat.__name__] == 1:
                    activated_feats.append(feat)
            except KeyError:
                raise KeyError(f"Feature '{feat.__name__}' does not exist in given configuration")
        return activated_feats
     
    def _apply_featurizers(self, document:str) -> List[Feature]:
        """Applies featurizers to a document and returns a list of features for a single document"""
        doc = make_document(document, self.nlp)
        features = []
        for featurizer in self._config:
            feature = featurizer(doc)
            counts = feature.feature_counts
            vector = feature.counts_to_vector()

            features.append(feature)
            feature_logger(featurizer.__name__, f"{counts}\n{vector}\n\n") 

        return features

    def _concat_vectors(self, features:List[Feature]) -> np.ndarray:
        """Concatenates a list of vectorized feature counts"""
        return np.concatenate([feat.counts_to_vector() for feat in features])
    
    def _get_all_feature_names(self, features:List[Feature]) -> List[str]:
        """Gets all feature names from a list of document features"""
        feature_names = []
        for feat in features:
            feature_names.extend(feat.get_feature_names())
        return feature_names
        
    def create_vector_df(self, documents:List[str]) -> pd.DataFrame:
        """Applies featurizers to all documents and stores the resulting matrix as a dataframe"""
        all_vectors = []
        feature_names = None
        for document in documents:
            doc_features = self._apply_featurizers(document)
            all_vectors.append(self._concat_vectors(doc_features))
            
            if feature_names is None:
                # hacky way of doing this, but it works well
                feature_names = self._get_all_feature_names(doc_features)  
             
        df = pd.DataFrame(np.vstack(all_vectors), columns=feature_names)
        return df
    
    def _remove_emojis(documents:List[str]) -> List[str]:
        """Removes emojis. Needed because spacy dependency parser hates emojis."""
        return list(map(lambda x: demoji.replace(x, "")))
    
    def _doc_generator(self, documents:List[str]) -> Generator[Doc, None, None]:
        """Converts a list of strings into a spacy Doc generator"""
        return nlp.pipe(documents, disable=["ner", "lemmatizer"])
    
    def from_jsonlines(self, path:str):
        """
        Generate a dataframe given a jsonlines file containing authorIDs and documentID fields
        
        This is done to make processing data in the T&E format easier
        """
        assert path.endswith(".jsonl"), f"Invalid file path: '{path}'. File must be specified as a jsonlines file (.jsonl)."
        df = pd.read_json(path, lines=True)
        try:
            documents = df["fullText"]
            author_ids = df["authorIDs"]
            document_ids = df["documentID"]
        except KeyError:
            raise KeyError("Specified jsonlines file missing one or more fields: 'fullText', 'authorIDs', 'documentID'")
        
        

    def from_document_list(self, documents:List[str]):
        """Generate a dataframe given a list of documents."""
        
        docs_demojified = self._remove_emojis(documents)
        
        for i, doc in enumerate(docs):
            pass



if __name__ == "__main__":
    
    print(nlp.pipeline)