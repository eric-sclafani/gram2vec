import os
import sys

# Set the language to Russian
os.environ["LANGUAGE"] = "ru"

# Add the parent directory to sys.path to import gram2vec
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from gram2vec
from gram2vec._load_vocab import vocab

def test_russian_vocab():
    """Test that Russian vocabulary is loaded correctly"""
    print("Testing Russian vocabulary loading:")
    
    # Test the dependency labels
    dep_labels = vocab.get("dep_labels")
    print(f"\nDependency labels ({len(dep_labels)} items):")
    print(", ".join(dep_labels[:10]) + "..." if len(dep_labels) > 10 else ", ".join(dep_labels))
    
    # Test the morph tags
    morph_tags = vocab.get("morph_tags")
    print(f"\nMorphological tags ({len(morph_tags)} items):")
    print(", ".join(morph_tags[:10]) + "..." if len(morph_tags) > 10 else ", ".join(morph_tags))
    
    # Test function words
    func_words = vocab.get("func_words")
    print(f"\nFunction words ({len(func_words)} items):")
    print(", ".join(func_words[:10]) + "..." if len(func_words) > 10 else ", ".join(func_words))
    
    # Test sentences patterns (from matcher)
    sentences = vocab.get("sentences")
    print(f"\nSentence patterns ({len(sentences)} items):")
    print(", ".join(sentences))

if __name__ == "__main__":
    test_russian_vocab() 