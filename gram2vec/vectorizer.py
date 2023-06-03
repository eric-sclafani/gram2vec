
from collections import Counter
import demoji
from nltk import bigrams
from nltk import FreqDist
import numpy as np 
import pandas as pd
import os
from typing import Optional, Tuple, List, Dict, Callable

from load_spacy import nlp
from generate_non_static_vocab import generate_non_static_vocab

# ~~~ Vocab ~~~

def load_from_txt(path:str) -> Tuple[str]:
    with open (path, "r") as fin:
        return tuple(map(lambda x: x.strip("\n"), fin.readlines()))
    
def vocab_loader() -> Dict[str, Tuple[str]]:
    """Loads in all feature vocabs. For any new vocabs, add them to this function"""
    static = "vocab/static/"
    non_static = "vocab/non_static/"
    return {
        "pos_unigrams": load_from_txt(static+"pos_unigrams.txt"),
        "pos_bigrams": load_from_txt(non_static+"pos_bigrams.txt"),
        "func_words": load_from_txt(static+"func_words.txt"),
        "punctuation": load_from_txt(static+"punctuation.txt"),
        "letters": load_from_txt(static+"letters.txt"),
        "common_emojis":load_from_txt(static+"common_emojis.txt"),
        "dep_labels": load_from_txt(static+"dep_labels.txt"),
        "mixed_bigrams":load_from_txt(non_static+"mixed_bigrams.txt"),
        "morph_tags":load_from_txt(static+"morph_tags.txt")
    }
    
VOCABS = vocab_loader()  
   
#~~~ Features ~~~

REGISTERD_FEATURES = {}

class Feature:
    """Encapsulates a feature counting function. When the function is called, normalization is applied to the counted features"""
    def __init__(self, func:Callable):
        self.func = func    
        self.name = func.__name__
        
    def __call__(self, doc, vocab):
        counted_features = self.func(doc)
        all_counts = self._include_zero_vocab_counts(counted_features, vocab)
        return self._normalize(all_counts)
    
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
    
    def _get_sum(self, counts:np.ndarray) -> int:
        """Gets sum of counts. Accounts for possible zero counts"""
        return sum(counts) if sum(counts) > 0 else 1

    def _normalize(self, counts:pd.Series) -> pd.Series:
        """Normalizes all counts"""
        return counts / self._get_sum(counts)
        

@Feature.register
def pos_unigrams(doc) -> Feature:
    return Counter(doc._.pos_tags)
    
@Feature.register
def pos_bigrams(doc) -> Feature:
    return Counter(doc._.pos_bigrams)

@Feature.register
def func_words(doc) -> Feature:
    return Counter([token for token in doc.tokens if token in VOCABS["func_words"]])
 
@Feature.register
def punctuation(doc) -> Feature:
    vocab = VOCABS["punctuation"]
    return Counter([punc for token in doc.tokens for punc in token if punc in vocab])

@Feature.register
def letters(doc) -> Feature:
    vocab = VOCABS["letters"]
    return Counter([letter for token in doc.tokens for letter in token if letter in vocab])

@Feature.register
def dep_labels(doc) -> Feature:
    return Counter([dep for dep in doc.dep_labels])

@Feature.register
def mixed_bigrams(doc) -> Feature:
    return Counter(doc._.mixed_bigrams)

@Feature.register
def morph_tags(doc) -> Feature:
    return Counter(doc._.morph_tags)


#! MAJOR WIP
# @Feature.register
# def common_emojis(doc) -> Feature:
    
#     vocab = load_vocab("vocab/static/common_emojis.txt")
#     extract_emojis = demoji.findall_list(doc.text, desc=False)
#     doc_emoji_counts = Counter(filter(lambda x: x in vocab, extract_emojis))
#     all_emoji_counts = add_zero_vocab_counts(vocab, doc_emoji_counts)
    
#     return Feature("Emoji", all_emoji_counts, len(doc.tokens))

#! DISABLED FOR NOW (only used for experimentation, NOT feature extraction)
# def embedding_vector(doc) -> Feature:
#     """spaCy word2vec (or glove?) document embedding"""
#     embedding = {"embedding_vector" : doc.spacy_doc.vector}
#     return Feature(None, embedding)

#! MAJOR WIP
# @Feature.register
# def document_stats(doc) -> Feature:
#     words = doc.words #!doc.words causing crash
    
#     #! GETTING SPLIT INTO SEPARATE VECTORS
#     #! REPLACE ALL STATISTICAL CALCULATIONS WITH BUILT IN
    
    
#     doc_statistics = {"short_words" : len([1 for word in words if len(word) < 5])/len(words) if len(words) else 0, 
#                       "large_words" : len([1 for word in words if len(word) > 4])/len(words) if len(words) else 0,
#                       "word_len_avg": np.mean([len(word) for word in words]),
#                       "word_len_std": np.std([len(word) for word in words]),
#                       "sent_len_avg": np.mean([len(sent) for sent in doc.sentences]),
#                       "sent_len_std": np.std([len(sent) for sent in doc.sentences]),
#                       "hapaxes"     : len(FreqDist(words).hapaxes())/len(words) if len(words) else 0}
#     return Feature("Document statistic", doc_statistics)


# ~~~ Processing ~~~


#! need a refresh parameter

     
class GrammarVectorizer:
    
  
    
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
    
    def _doc_generator(self, documents:List[str]):
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
    s = "it was Jane’s car that got stolen last night."
    doc = nlp(s)
    
    
    print(pos_bigrams(doc, vocabs["pos_bigrams"]))