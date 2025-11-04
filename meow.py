from padie_extended.language_detector import LanguageDetector

def test_predict_english(detector):
    """Test that English text is correctly identified."""
    text = "Good morning, how are you doing today?"
    result = detector.predict(text)
    print(result["all_scores"])

detector = LanguageDetector()
test_predict_english(detector)