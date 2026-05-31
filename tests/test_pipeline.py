import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from nlp_pipeline import NLPipeline, TextPreprocessor

def test_preprocessor():
    p = TextPreprocessor(lowercase=True)
    assert "hello" in p.preprocess_text("HELLO world")

def test_pipeline_init():
    p = NLPipeline()
    assert p is not None

if __name__ == "__main__":
    test_preprocessor()
    test_pipeline_init()
    print("Basic tests passed!")
