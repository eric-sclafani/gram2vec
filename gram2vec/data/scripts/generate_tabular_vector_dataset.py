
#! NEEDS TO BE FORMATTED AND FIXED

@dataclass
class AuthorEntry:
    author_id:str
    discourse_type:str
    fixed_text:str

def iter_author_jsonls(author_files_dir:str) -> str:
    """Yields each {author_id}.jsonl from a given dir"""
    for author_file in Path(author_files_dir).glob("*.jsonl"):
        yield author_file
        
def iter_author_entries(author_file):
    """Yields each JSON object from an {author_id}.jsonl file"""
    with jsonlines.open(author_file) as author_entries:
        for entry in author_entries:
            yield entry
            
def get_all_entries(data_dir:str) -> list[AuthorEntry]:
    """Extracts author file entries as AuthorEntry objects and aggregates into one list"""
    all_entries = []
    for author_file in iter_author_jsonls(data_dir):
        for entry in iter_author_entries(author_file):
            all_entries.append(AuthorEntry(entry["author_id"], entry["discourse_type"], entry["fixed_text"]))
    return all_entries

data_dir = "data/pan22/preprocessed/"
all_entries = get_all_entries(data_dir)

documents = [entry.fixed_text for entry in all_entries]
authors = [entry.author_id for entry in all_entries]
discourse_types = [entry.discourse_type for entry in all_entries]

g2v = GrammarVectorizer()

feature_vectors = g2v.vectorize_episode(documents)


def get_vocab(path) -> list[str]:
    with open(path, "r") as fin:
        return fin.read().strip().split("\n")

pos_unigrams  = get_vocab("vocab/static/pos_unigrams.txt")
pos_bigrams   = get_vocab("vocab/non_static/pos_bigrams/pan/pos_bigrams.txt")
func_words    = get_vocab("vocab/static/function_words.txt")
punc          = get_vocab("vocab/static/punc_marks.txt")
letters       = get_vocab("vocab/static/letters.txt")
common_emojis = get_vocab("vocab/static/common_emojis.txt")
doc_stats     = ["short_words", "large_words", "word_len_avg", "word_len_std", "sent_len_avg", "sent_len_std", "hapaxes"]
deps          = get_vocab("vocab/static/dep_labels.txt")
mixed_bigrams = get_vocab("vocab/non_static/mixed_bigrams/pan/mixed_bigrams.txt")

all_features = pos_unigrams + pos_bigrams + func_words + punc + letters + common_emojis + doc_stats + deps + mixed_bigrams

def convert_feature_name(feature:str, seen_i, seen_a, seen_X) -> str:
    """
    Hard coded way of making certain conflicting feature names unique.
    Needs to be done to ensure each feature in data viz dataset is unique
    """
    if feature == "i": 
        feature = "i (func_word)" if not seen_i else "i (letter)"
            
    elif feature == "a":
        feature = "a (func_word)" if not seen_a else "a (letter)"
        
    elif feature == "X":
        feature = "X (pos_unigram)" if not seen_X else "X (letter)"
        
    return feature
        

def make_feature_to_counts_map(all_features:list[str]) -> dict[str,list]:
    """
    Maps each feature to an empty list. Accounts for DIFFERENT features with the SAME label
    
    i.e. some distinct features have the same labels ("i", "a", "X"), so for data visualization purposes,
    they need to be renamed to be distinct. This DOES NOT affect the vectors in any way. 
    
    The conditionals here follow the same concatenation order as all_features
    """
    seen_i, seen_a, seen_X = False, False, False
    count_dict = {}
    for feature in all_features:
        if feature == "i":
            feature = convert_feature_name(feature, seen_i, seen_a, seen_X)
            seen_i = True
            
        if feature == "a":
            feature = convert_feature_name(feature, seen_i, seen_a, seen_X)
            seen_a = True
            
        if feature == "X":
            feature = convert_feature_name(feature, seen_i, seen_a, seen_X)
            seen_X = True
            
        count_dict[feature] = []
    return count_dict

            
def populate_feature_to_counts_map(all_features:list[str], feature_vectors:list) -> dict[str,list[int]]:
    """
    Populates the feature_to_count dict. Accounts for DIFFERENT features with the SAME label
    
    For every feature's count_dict, append the feature name's count number to 
    corresponding list in feats_to_counts
    """
    feats_to_counts = make_feature_to_counts_map(all_features)
    seen_i, seen_a, seen_X = False, False, False
    
    for feature in feature_vectors:
        for count_dict in feature.count_map.values():
            for feat_name, count in count_dict.items():
                
                if feat_name == "i":
                    feat_name = convert_feature_name(feat_name, seen_i, seen_a, seen_X)
                    seen_i = True
                    
                if feat_name == "a":
                    feat_name = convert_feature_name(feat_name, seen_i, seen_a, seen_X)
                    seen_a = True
                    
                if feat_name == "X":
                    feat_name = convert_feature_name(feat_name, seen_i, seen_a, seen_X)
                    seen_X = True
                       
                feats_to_counts[str(feat_name)].append(count)
        seen_i, seen_a, seen_X = False, False, False # reset flags for every count_dict
            
    return feats_to_counts

features_to_count_lists = populate_feature_to_counts_map(all_features, feature_vectors)



df = pd.DataFrame(features_to_count_lists)
df.insert(0, "author_id", authors)
df.insert(1, "discourse_type", discourse_types)


try:
    os.chdir("../../cse564/project2/data/")
except:pass

df.to_csv("pan22_features.csv", index=None)