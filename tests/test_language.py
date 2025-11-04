import pytest
from language_detector import LanguageDetector

# Load the model ONCE globally
# This ensures the 1GB model is initialized a single time for all tests
@pytest.fixture(scope="session")
def detector():
    """Initialize LanguageDetector once per test session."""
    return LanguageDetector()


def test_detect_single_text(detector):
    """Test detection for a single input string."""
    text = "How far?"
    result = detector.predict(text)
    assert isinstance(result, dict)
    assert "language" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], float)


def test_confidence_threshold(detector):
    """Ensure low-confidence predictions are handled correctly."""
    uncertain_text = "Blahblahbluh"
    result = detector.predict(uncertain_text, threshold=0.9)
    assert result["low_confidence"] is True
    assert result["language"] == "uncertain"


def test_batch_prediction(detector):
    """Test batch prediction capability."""
    texts = ["How you dey?", "E kaaro", "Nno", "Ina kwana"]
    results = detector.predict_batch(texts)
    assert isinstance(results, list)
    assert len(results) == len(texts)
    for res in results:
        assert "language" in res
        assert "confidence" in res


def test_threshold_update(detector):
    """Ensure threshold setter works as expected."""
    detector.set_threshold(0.7)
    assert detector.confidence_threshold == 0.7

    # Invalid threshold should raise an error
    with pytest.raises(ValueError):
        detector.set_threshold(2)