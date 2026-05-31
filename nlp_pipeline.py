#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2024 Vincent Onyecherem Ikenna

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import re
import string
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline

# ============ 1. TEXT PREPROCESSING MODULE ============

class TextPreprocessor:
    """
    Comprehensive text preprocessing with multiple cleaning options
    """
    def __init__(self, 
                 lowercase=True,
                 remove_punctuation=True,
                 remove_numbers=False,
                 remove_stopwords=False,
                 stem=False,
                 lemmatize=False,
                 min_word_length=2,
                 custom_stopwords=None):
        
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        
        # Default English stopwords
        self.stopwords = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'without', 'after', 'before', 'upon', 'between',
            'into', 'through', 'during', 'to', 'from', 'up', 'down', 'in', 'out',
            'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'that', 'these', 'those', 'very', 'just', 'but', 'do', 'does', 'doing',
            'should', 'could', 'would', 'can', 'may', 'might', 'must'
        ])
        
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        
        # Simple stemmer (Porter Stemmer algorithm approximation)
        self.stemmer = self._simple_stemmer
        
    def _simple_stemmer(self, word):
        """Basic stemmer implementation"""
        word = word.lower()
        suffixes = ['ing', 'ly', 'ed', 'ies', 'es', 's', 'ness', 'ment', 'tion']
        
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break
                
        if word.endswith('e') and len(word) > 3:
            word = word[:-1]
            
        return word
    
    def _simple_lemmatizer(self, word):
        """Basic lemmatizer implementation"""
        # Common irregular forms
        irregular = {
            'am': 'be', 'are': 'be', 'is': 'be', 'was': 'be', 'were': 'be',
            'has': 'have', 'have': 'have', 'had': 'have',
            'does': 'do', 'do': 'do', 'did': 'do',
            'goes': 'go', 'going': 'go', 'went': 'go',
            'runs': 'run', 'running': 'run', 'ran': 'run',
            'eats': 'eat', 'eating': 'eat', 'ate': 'eat',
            'cats': 'cat', 'dogs': 'dog', 'children': 'child',
            'mice': 'mouse', 'feet': 'foot', 'teeth': 'tooth'
        }
        
        if word in irregular:
            return irregular[word]
        
        # Simple suffix removal for lemmatization
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'
        elif word.endswith('es') and len(word) > 3:
            return word[:-2]
        elif word.endswith('s') and len(word) > 2:
            return word[:-1]
        
        return word
    
    def preprocess_text(self, text):
        """Main preprocessing pipeline for a single text"""
        if not isinstance(text, str):
            text = str(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Tokenization (simple whitespace splitting)
        tokens = text.split()
        
        # Apply preprocessing to each token
        processed_tokens = []
        for token in tokens:
            # Remove short words
            if len(token) < self.min_word_length:
                continue
                
            # Remove stopwords
            if self.remove_stopwords and token in self.stopwords:
                continue
            
            # Stemming
            if self.stem:
                token = self.stemmer(token)
            
            # Lemmatization
            if self.lemmatize:
                token = self._simple_lemmatizer(token)
            
            processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def preprocess_corpus(self, texts):
        """Preprocess a list of texts"""
        return [self.preprocess_text(text) for text in texts]

# ============ 2. FEATURE EXTRACTION MODULE ============

class FeatureExtractor:
    """
    Extract various features from text data
    """
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 2)):
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.feature_names = None
        
    def fit(self, texts):
        """Fit the vectorizer to the corpus"""
        if self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                lowercase=False  # Already preprocessed
            )
        elif self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                lowercase=False
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self
    
    def transform(self, texts):
        """Transform texts into feature vectors"""
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform the texts"""
        self.fit(texts)
        return self.transform(texts)

class CustomFeatureExtractor:
    """
    Build custom features beyond bag-of-words
    """
    @staticmethod
    def extract_length_features(text):
        """Extract text length statistics"""
        words = text.split()
        return {
            'char_count': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'unique_words_ratio': len(set(words)) / len(words) if words else 0
        }
    
    @staticmethod
    def extract_linguistic_features(text):
        """Extract linguistic features"""
        words = text.split()
        features = {}
        
        # Capitalization features
        features['cap_words_ratio'] = sum(1 for w in words if w[0].isupper()) / len(words) if words else 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['period_count'] = text.count('.')
        
        # Special characters
        features['hashtag_count'] = text.count('#')
        features['mention_count'] = text.count('@')
        
        # Sentiment markers
        positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like'])
        negative_words = set(['bad', 'terrible', 'awful', 'hate', 'dislike', 'poor', 'worst'])
        
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        
        features['sentiment_score'] = pos_count - neg_count
        features['sentiment_ratio'] = pos_count / (neg_count + 1)
        
        return features
    
    @staticmethod
    def extract_all_features(text):
        """Combine all feature types"""
        features = {}
        features.update(CustomFeatureExtractor.extract_length_features(text))
        features.update(CustomFeatureExtractor.extract_linguistic_features(text))
        return features

# ============ 3. CUSTOM TRANSFORMER FOR SKLEARN PIPELINE ============

class TextPreprocessorTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible text preprocessor"""
    def __init__(self, **preprocessor_kwargs):
        self.preprocessor = TextPreprocessor(**preprocessor_kwargs)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.preprocessor.preprocess_corpus(X)

class FeatureExtractorTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible feature extractor"""
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 2)):
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.extractor = FeatureExtractor(method, max_features, ngram_range)
    
    def fit(self, X, y=None):
        self.extractor.fit(X)
        return self
    
    def transform(self, X):
        return self.extractor.transform(X)

# ============ 4. MAIN NLP PIPELINE CLASS ============

class NLPipeline:
    """
    Complete NLP pipeline combining preprocessing, feature extraction, and modeling
    """
    def __init__(self, 
                 preprocess_kwargs=None,
                 feature_method='tfidf',
                 max_features=5000,
                 classifier='logistic_regression',
                 classifier_kwargs=None):
        
        # Default preprocessing kwargs
        default_preprocess = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_numbers': False,
            'remove_stopwords': True,
            'stem': False,
            'lemmatize': False,
            'min_word_length': 2
        }
        
        if preprocess_kwargs:
            default_preprocess.update(preprocess_kwargs)
        
        self.preprocessor = TextPreprocessor(**default_preprocess)
        self.feature_extractor = FeatureExtractor(
            method=feature_method,
            max_features=max_features
        )
        
        # Initialize classifier
        if classifier_kwargs is None:
            classifier_kwargs = {}
        
        if classifier == 'logistic_regression':
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=42,
                **classifier_kwargs
            )
        elif classifier == 'naive_bayes':
            self.classifier = MultinomialNB(**classifier_kwargs)
        else:
            raise ValueError(f"Unknown classifier: {classifier}")
        
        self.is_fitted = False
    
    def preprocess(self, texts):
        """Preprocess raw texts"""
        if isinstance(texts, str):
            texts = [texts]
        return self.preprocessor.preprocess_corpus(texts)
    
    def extract_features(self, texts):
        """Extract features from preprocessed texts"""
        if not self.feature_extractor.vectorizer:
            self.feature_extractor.fit(texts)
        return self.feature_extractor.transform(texts)
    
    def fit(self, X, y):
        """Fit the entire pipeline"""
        print("Preprocessing texts...")
        X_preprocessed = self.preprocess(X)
        
        print("Extracting features...")
        X_features = self.extract_features(X_preprocessed)
        
        print("Training classifier...")
        self.classifier.fit(X_features, y)
        
        self.is_fitted = True
        print("Pipeline training complete!")
        return self
    
    def predict(self, X):
        """Predict labels for new texts"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        X_preprocessed = self.preprocess(X)
        X_features = self.extract_features(X_preprocessed)
        return self.classifier.predict(X_features)
    
    def predict_proba(self, X):
        """Predict probabilities for new texts"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        X_preprocessed = self.preprocess(X)
        X_features = self.extract_features(X_preprocessed)
        return self.classifier.predict_proba(X_features)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the pipeline on test data"""
        y_pred = self.predict(X_test)
        
        print("\n=== Evaluation Results ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

# ============ 5. DEMO AND TESTING ============

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    texts = [
        "I love this product, it's amazing!",
        "Great quality, highly recommended.",
        "Terrible experience, waste of money.",
        "Worst purchase ever, very disappointed.",
        "Excellent service, will buy again.",
        "Not worth the price, very poor quality.",
        "Perfect! Exactly what I needed.",
        "Disappointing, didn't meet expectations.",
        "Fantastic! Best product ever.",
        "Would not recommend, very bad.",
        "Good value for money, decent quality.",
        "Average product, nothing special.",
        "Outstanding performance, highly satisfied.",
        "Poor customer service, terrible support.",
        "Amazing features, love it!",
    ]
    
    labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1: positive, 0: negative
    
    return texts, labels

def main():
    """Main demonstration function"""
    print("="*50)
    print("NLP PIPELINE FROM SCRATCH")
    print("="*50)
    
    # Load sample data
    texts, labels = create_sample_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Create and train pipeline
    pipeline = NLPipeline(
        preprocess_kwargs={
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': True,
            'stem': False
        },
        feature_method='tfidf',
        max_features=100,
        classifier='logistic_regression'
    )
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    results = pipeline.evaluate(X_test, y_test)
    
    # Test prediction on new text
    new_texts = [
        "This is absolutely wonderful!",
        "Terrible product, don't buy it",
        "It's okay, nothing special"
    ]
    
    print("\n=== Predictions on New Texts ===")
    predictions = pipeline.predict(new_texts)
    probabilities = pipeline.predict_proba(new_texts)
    
    for text, pred, prob in zip(new_texts, predictions, probabilities):
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = prob[pred]
        print(f"\nText: '{text}'")
        print(f"Sentiment: {sentiment} (confidence: {confidence:.3f})")
    
    # Demonstrate custom feature extraction
    print("\n" + "="*50)
    print("CUSTOM FEATURE EXTRACTION EXAMPLE")
    print("="*50)
    
    sample_text = "This is an amazing product! I love it so much!!"
    custom_features = CustomFeatureExtractor.extract_all_features(sample_text)
    
    print(f"Text: '{sample_text}'")
    print("Extracted features:")
    for key, value in custom_features.items():
        print(f"  {key}: {value}")

# ============ 6. ADVANCED FEATURES ============

class BatchProcessor:
    """Process large batches of text efficiently"""
    def __init__(self, pipeline, batch_size=100):
        self.pipeline = pipeline
        self.batch_size = batch_size
    
    def predict_batch(self, texts, show_progress=True):
        """Predict in batches to manage memory"""
        predictions = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_preds = self.pipeline.predict(batch)
            predictions.extend(batch_preds)
            
            if show_progress and (i // self.batch_size) % 10 == 0:
                print(f"Processed {i+len(batch)}/{len(texts)} texts")
        
        return predictions

class ModelPersistence:
    """Save and load trained pipelines"""
    import joblib
    
    @staticmethod
    def save_pipeline(pipeline, filepath):
        import joblib
        joblib.dump(pipeline, filepath)
        print(f"Pipeline saved to {filepath}")
    
    @staticmethod
    def load_pipeline(filepath):
        import joblib
        pipeline = joblib.load(filepath)
        print(f"Pipeline loaded from {filepath}")
        return pipeline

# ============ 7. USAGE EXAMPLE ============

if __name__ == "__main__":
    main()
    
    # Example of using scikit-learn pipeline wrapper
    print("\n" + "="*50)
    print("SKLEARN PIPELINE WRAPPER EXAMPLE")
    print("="*50)
    
    # Create sklearn-compatible pipeline
    sklearn_pipeline = SklearnPipeline([
        ('preprocessor', TextPreprocessorTransformer(
            lowercase=True,
            remove_punctuation=True,
            remove_stopwords=True
        )),
        ('vectorizer', TfidfVectorizer(max_features=100, ngram_range=(1,2))),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    # Load data
    texts, labels = create_sample_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    
    # Train sklearn pipeline
    sklearn_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = sklearn_pipeline.predict(X_test)
    print(f"Sklearn Pipeline Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    