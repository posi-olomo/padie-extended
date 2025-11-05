from padie_extended.language_detector import LanguageDetector

def test_predict_english(detector):
    """Test that English text is correctly identified."""
    text = "Good morning, how are you doing today?"
    result = detector.predict(text)
    print(result["all_scores"])



detector = LanguageDetector()
test_predict_english(detector)

def test_batch_prediction(detector):
    """Test batch prediction capability."""
    texts = ["How you dey?", "E kaaro", "Nno", "Ina kwana"]
    results = detector.predict_batch(texts)
    print(results)

test_batch_prediction(detector)