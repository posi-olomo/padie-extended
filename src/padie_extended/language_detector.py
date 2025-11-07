from transformers import pipeline
import torch
from typing import Dict, Optional, Union

class LanguageDetector:
    """
    Detects Nigerian languages including English, Pidgin, Yoruba, Hausa and Igbo

    Example:
        >>> detector = LanguageDetector()
        >>> result = detector.predict("How you dey?")
        >>> print(result['language'])
        'pidgin'
    """
    def __init__(
        self,
        model_path: Optional[str] = None, 
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the language detector

        Args:
            model_path: Path to the directory.
            confidence_threshold: Minimum confidence (0-1) for valid predictions.
                                    Predictions below this return a low confidence warning.
        """
        if model_path is None:
            # The model_path is my uploaded HuggingFace Model
            model_path = "posi-olomo/padie-extended"
        
        self.confidence_threshold = confidence_threshold

        # Initialize pipeline with auto device detection
        device = 0 if torch.cuda.is_available() else -1

        self.pipeline = pipeline(
            "text-classification",
            model = model_path,
            device = device
        )

    def predict(
            self,
            text: str,
            threshold: Optional[float] = None
    ) -> Dict[str, Union[str, float, bool]]:
        """
        Predict the language of input text.

        Args:
            text: Input text to classify
            threshold: Override the default confidence threshold for this prediction

        Returns:
            Dictionary with:
                - language: Predicted language or 'uncertain'
                - all_scores: all_scores
                - confidence: Confidence score (0-1)
                - low_confidence: Boolean indicating if confidence is below threshold
                - message: Warning message if confidence is low
                - raw_prediction: What would have been predicted
        
        Example:
            >>> result = detector.predict("Hello there")
            >>> print(result)
            {
                'language': 'uncertain',
                'all_scores': {
                            'english': 0.3021,
                            'pidgin': 0.2045,
                            'hausa': 0.2450,
                            'igbo': 0.2262,
                            'yoruba': 0.0222
                            },
                'confidence': 0.45,
                'low_confidence': True,
                'message': 'Low confidence prediction. This might not be a Nigerian language.'
                'raw_prediction': english
            }
        """
        # Handle empty input
        if not text or not text.strip():
            return {
                'language': 'unkown',
                'confidence': 0.0,
                'low_confidence': True,
                'message': 'Empty input text provided'
            }
        
        # Use custom threshold if provided, otherwise use default
        threshold_to_use = threshold if threshold is not None else self.confidence_threshold 

        # Get prediction
        result = self.pipeline(text, top_k = None)

        confidence = result[0]['score']
        predicted_language = result[0]['label']
        # a dictionary that sets key as the labels and value as the score
        all_scores = {pred['label']:pred['score'] for pred in result}
        
        # Check if confidence is below threshold
        if confidence < threshold_to_use:
            return {
                'language': 'uncertain',
                'confidence': confidence,
                'all_scores': all_scores,
                'low_confidence': True,
                'message': f'Low confidence prediction ({confidence:.2%}). This might not be a Nigerian language.',
                'raw_prediction': predicted_language 
            }

        # High confidence prediction 
        return {
            'language': predicted_language,
            'confidence': confidence,
            'all_scores': all_scores,
            'low_confidence': False
            }
    
    def predict_batch(
            self,
            texts: list,
            threshold: Optional[float] = None 
    ) -> list:
        """
        Predict languages for multiple texts.

        Args:
            texts: List of input texts
            threshold: Override the default confidence threshold 

        Returns:
            List of prediction dictionaries
        """
        threshold_to_use = threshold if threshold is not None else self.confidence_threshold

        # Get batch predictions
        raw_results = self.pipeline(texts, top_k = None)
        
        # Process each result
        return [
            {
                'language': 'uncertain',
                'confidence': result[0]['score'],
                'all_scores': {pred['label']:pred['score'] for pred in result},
                'low_confidence': True,
                'message': f"Low confidence prediction ({result[0]['score']:.2%}). This might not be a Nigerian language.",
                'raw_prediction': result[0]['label']
            } if result[0]['score'] < threshold_to_use else {
                'language': result[0]['label'],
                'confidence': result[0]['score'],
                'all_scores': {pred['label']:pred['score'] for pred in result},
                'low_confidence': False
            }
            for result in raw_results
        ]

    def set_threshold(self, threshold:float):
        """
        Update the confidence threshold

        Args:
            threshold: New confidence threshold (0-1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.confidence_threshold = threshold

    def __repr__(self) -> str:
        return (
            f"languageDetector("
            f"threshold = {self.confidence_threshold},"
            f"device = {'cuda' if torch.cuda.is_available() else 'cpu'})"
        )