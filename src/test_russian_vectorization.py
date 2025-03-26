import os
import sys

# Set the language to Russian
os.environ["LANGUAGE"] = "ru"
# Set appropriate spaCy model for Russian
os.environ["SPACY_MODEL"] = "ru_core_news_lg"

# Add the parent directory to sys.path to import gram2vec
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from gram2vec
from gram2vec import vectorizer

def test_russian_vectorization():
    """Test that vectorization works correctly with Russian text"""
    print("Testing Russian text vectorization:")
    
    # Sample Russian text
    russian_text = [
        "Мой дядя самых честных правил, когда не в шутку занемог, он уважать себя заставил и лучше выдумать не мог.",
        "Москва и Петербург - два главных города России, они отличаются по своей истории и архитектуре."
    ]
    
    print(f"\nInput texts:\n1. {russian_text[0]}\n2. {russian_text[1]}")
    
    # Vectorize the sample text
    print("\nVectorizing Russian text...")
    result = vectorizer.from_documents(russian_text)
    
    print(f"\nResult shape: {result.shape}")
    
    # Print some feature samples
    print("\nSample features:")
    for prefix in ["dep_labels:", "morph_tags:", "func_words:"]:
        # Get a few columns that start with this prefix
        sample_cols = [col for col in result.columns if col.startswith(prefix)][:3]
        if sample_cols:
            print(f"\n{prefix[:-1]} samples:")
            print(result[sample_cols])

if __name__ == "__main__":
    test_russian_vectorization() 