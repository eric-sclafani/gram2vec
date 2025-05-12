"""
This module provides functionality to locate specific gram2vec features within a text.
It finds spans of text where specific features occur.
"""

from typing import List, Dict, Tuple, Optional, Union
import re
import spacy
import demoji
from dataclasses import dataclass

from ._load_spacy import nlp
from ._load_vocab import vocab
from .vectorizer import Gram2VecDocument, _remove_emojis


@dataclass
class FeatureSpan:
    """Represents a text span where a specific feature occurs"""
    feature_type: str
    feature_name: str
    text: str
    start_char: int
    end_char: int
    
    def __repr__(self) -> str:
        return f"{self.feature_type}:{self.feature_name} at char [{self.start_char}:{self.end_char}]: '{self.text}'"


class FeatureLocator:
    """
    Locates specific gram2vec features within text spans.
    
    This class allows finding where specific features (like POS tags, function words, etc.)
    occur within a text document.
    """
    
    def __init__(self):
        """Initializes the FeatureLocator."""
        self.feature_handlers = {
            "pos_unigrams": self._locate_pos_unigrams,
            "pos_bigrams": self._locate_pos_bigrams,
            "func_words": self._locate_func_words,
            "punctuation": self._locate_punctuation,
            "letters": self._locate_letters,
            "dep_labels": self._locate_dep_labels,
            "morph_tags": self._locate_morph_tags,
            "sentences": self._locate_sentences,
            "emojis": self._locate_emojis
        }
    
    def locate_feature(self, text: str, feature_name: str) -> List[FeatureSpan]:
        """
        Locates all occurrences of a specific feature in the text.
        
        Args:
            text: The input text to search in
            feature_name: The full feature name in format "feature_type:feature_value"
                          (e.g., "pos_unigrams:NOUN", "func_words:the")
        
        Returns:
            A list of FeatureSpan objects representing each occurrence of the feature
        """
        if ":" not in feature_name:
            raise ValueError("Feature name must be in format 'feature_type:feature_value'")
            
        feature_type, feature_value = feature_name.split(":", 1)
        
        if feature_type not in self.feature_handlers:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Process the document
        doc = self._process_document(text)
        
        # Use the appropriate handler to locate the feature
        return self.feature_handlers[feature_type](doc, feature_value)
    
    def _process_document(self, text: str) -> Gram2VecDocument:
        """Process a text document and return the Gram2VecDocument"""
        # Process the raw text directly without removing emojis
        spacy_doc = nlp(text)
        return Gram2VecDocument(text, spacy_doc, len(spacy_doc), len(list(spacy_doc.sents)))
    
    def _locate_pos_unigrams(self, doc: Gram2VecDocument, feature_value: str) -> List[FeatureSpan]:
        """Locate all tokens with the specified POS tag"""
        spans = []
        for token in doc.doc:
            if token.pos_ == feature_value:
                spans.append(FeatureSpan(
                    feature_type="pos_unigrams",
                    feature_name=feature_value,
                    text=token.text,
                    start_char=token.idx,
                    end_char=token.idx + len(token.text)
                ))
        return spans
    
    def _locate_pos_bigrams(self, doc: Gram2VecDocument, feature_value: str) -> List[FeatureSpan]:
        """Locate all POS bigrams in the text"""
        spans = []
        pos1, pos2 = feature_value.split()
        
        # Handle special boundary symbols
        if pos1 == "BOS" and pos2 != "EOS":
            for sent in doc.doc.sents:
                if sent[0].pos_ == pos2:
                    spans.append(FeatureSpan(
                        feature_type="pos_bigrams",
                        feature_name=feature_value,
                        text=sent[0].text,
                        start_char=sent[0].idx,
                        end_char=sent[0].idx + len(sent[0].text)
                    ))
        elif pos2 == "EOS" and pos1 != "BOS":
            for sent in doc.doc.sents:
                if sent[-1].pos_ == pos1:
                    spans.append(FeatureSpan(
                        feature_type="pos_bigrams",
                        feature_name=feature_value,
                        text=sent[-1].text,
                        start_char=sent[-1].idx,
                        end_char=sent[-1].idx + len(sent[-1].text)
                    ))
        elif pos1 == "BOS" and pos2 == "EOS":
            # Single token sentences
            for sent in doc.doc.sents:
                if len(sent) == 1:
                    spans.append(FeatureSpan(
                        feature_type="pos_bigrams",
                        feature_name=feature_value,
                        text=sent.text,
                        start_char=sent[0].idx,
                        end_char=sent[0].idx + len(sent.text)
                    ))
        else:
            # Regular POS bigrams
            for sent in doc.doc.sents:
                for i in range(len(sent) - 1):
                    if sent[i].pos_ == pos1 and sent[i+1].pos_ == pos2:
                        text = sent[i].text + " " + sent[i+1].text
                        spans.append(FeatureSpan(
                            feature_type="pos_bigrams",
                            feature_name=feature_value,
                            text=text,
                            start_char=sent[i].idx,
                            end_char=sent[i+1].idx + len(sent[i+1].text)
                        ))
        return spans
    
    def _locate_func_words(self, doc: Gram2VecDocument, feature_value: str) -> List[FeatureSpan]:
        """Locate all occurrences of a specific function word"""
        func_words_vocab = vocab.get("func_words")
        spans = []
        
        if feature_value not in func_words_vocab:
            return spans
            
        for token in doc.doc:
            if token.text.lower() == feature_value.lower():
                spans.append(FeatureSpan(
                    feature_type="func_words",
                    feature_name=feature_value,
                    text=token.text,
                    start_char=token.idx,
                    end_char=token.idx + len(token.text)
                ))
        return spans
    
    def _locate_punctuation(self, doc: Gram2VecDocument, feature_value: str) -> List[FeatureSpan]:
        """Locate all occurrences of a specific punctuation mark"""
        spans = []
        for token in doc.doc:
            if token.text == feature_value and token.is_punct:
                spans.append(FeatureSpan(
                    feature_type="punctuation",
                    feature_name=feature_value,
                    text=token.text,
                    start_char=token.idx,
                    end_char=token.idx + len(token.text)
                ))
        return spans
    
    def _locate_letters(self, doc: Gram2VecDocument, feature_value: str) -> List[FeatureSpan]:
        """Locate all occurrences of a specific letter"""
        spans = []
        raw_text = doc.raw
        
        for match in re.finditer(re.escape(feature_value), raw_text):
            spans.append(FeatureSpan(
                feature_type="letters",
                feature_name=feature_value,
                text=feature_value,
                start_char=match.start(),
                end_char=match.end()
            ))
        return spans
    
    def _locate_dep_labels(self, doc: Gram2VecDocument, feature_value: str) -> List[FeatureSpan]:
        """Locate all tokens with the specified dependency label"""
        spans = []
        for token in doc.doc:
            if token.dep_ == feature_value:
                spans.append(FeatureSpan(
                    feature_type="dep_labels",
                    feature_name=feature_value,
                    text=token.text,
                    start_char=token.idx,
                    end_char=token.idx + len(token.text)
                ))
        return spans
    
    def _locate_morph_tags(self, doc: Gram2VecDocument, feature_value: str) -> List[FeatureSpan]:
        """Locate all tokens with the specified morphological tag"""
        spans = []
        for token in doc.doc:
            # Check if the specific morphological feature is present in this token
            for morph in token.morph:
                if morph == feature_value:
                    spans.append(FeatureSpan(
                        feature_type="morph_tags",
                        feature_name=feature_value,
                        text=token.text,
                        start_char=token.idx,
                        end_char=token.idx + len(token.text)
                    ))
                    break
        return spans
    
    def _locate_sentences(self, doc: Gram2VecDocument, feature_value: str) -> List[FeatureSpan]:
        """Locate all sentences that match a specific sentence pattern"""
        from .matcher import SyntaxRegexMatcher
        
        # Get language from environment or default to English
        import os
        language = os.environ.get("LANGUAGE", "en")
        
        # Create a matcher to find sentence patterns
        matcher = SyntaxRegexMatcher(language=language)
        
        # Find matches
        matches = matcher.match_document(doc.doc)
        
        # Filter matches by pattern name
        spans = []
        for match in matches:
            if match.pattern_name == feature_value:
                for sent in doc.doc.sents:
                    if sent.text == match.sentence:
                        spans.append(FeatureSpan(
                            feature_type="sentences",
                            feature_name=feature_value,
                            text=sent.text,
                            start_char=sent.start_char,
                            end_char=sent.end_char
                        ))
        
        return spans
    
    def _locate_emojis(self, doc: Gram2VecDocument, feature_value: str) -> List[FeatureSpan]:
        """Locate all occurrences of a specific emoji"""
        spans = []
        raw_text = doc.raw
        
        # Handle special OOV emoji case
        if feature_value == "OOV_emoji":
            emojis_vocab = vocab.get("emojis")
            all_emojis = demoji.findall(raw_text)
            
            for emoji, position in all_emojis.items():
                if emoji not in emojis_vocab:
                    # Convert position to integer
                    start_pos = raw_text.find(emoji)
                    spans.append(FeatureSpan(
                        feature_type="emojis",
                        feature_name="OOV_emoji",
                        text=emoji,
                        start_char=start_pos,
                        end_char=start_pos + len(emoji)
                    ))
        else:
            # Regular emoji lookup
            for match in re.finditer(re.escape(feature_value), raw_text):
                spans.append(FeatureSpan(
                    feature_type="emojis",
                    feature_name=feature_value,
                    text=feature_value,
                    start_char=match.start(),
                    end_char=match.end()
                ))
        
        return spans


def find_feature_spans(text: str, feature: str) -> List[FeatureSpan]:
    """
    Find all spans in the text where the specified feature occurs.
    
    Args:
        text: Input text to analyze
        feature: Feature name in format "feature_type:feature_value" 
                (e.g., "pos_unigrams:NOUN", "func_words:the")
    
    Returns:
        List of FeatureSpan objects representing occurrences of the feature in the text
    """
    locator = FeatureLocator()
    return locator.locate_feature(text, feature) 