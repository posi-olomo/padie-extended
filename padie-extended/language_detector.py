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
            # model_path = "models/full/language_detection"
            # The model_path is not hard coded as it is above because path is relative to wherever the user
            # runs their script from. This can easily lead to errors. 
            # You need the model_path to be relative to where the package is installed. 
            import os
            model_path = os.path.join(
                os.path.dirname(__file__),
                "models", "full", "language_detection"
            )
        
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
                - confidence: Confidence score (0-1)
                - low_confidence: Boolean indicating if confidence is below threshold
                - message: Warning message if confidence is low
                - raw_prediction: What would have been predicted
        
        Example:
            >>> result = detector.predict("Hello there")
            >>> print(result)
            {
                'language': 'uncertain',
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
        result = self.pipeline(text)[0]

        confidence = result['score']
        predicted_language = result['label']

        # Check if confidence is below threshold

        if confidence < threshold_to_use:
            return {
                'language': 'uncertain',
                'confidence': confidence,
                'low_confidence': True,
                'message': f'Low confidence prediction ({confidence:.2%}). This might not be a Nigerian language.',
                'raw_prediction': predicted_language 
            }

        # High confidence prediction 
        return {
            'language': predicted_language,
            'confidence': confidence,
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
        raw_results = self.pipeline(texts)
        
        # Process each result
        return [
            {
                'language': 'uncertain',
                'confidence': result['score'],
                'low_confidence': True,
                'message': f'Low confidence prediction ({result['score']:.2%}). This might not be a Nigerian language.'
                'raw_prediction': result['label']
            } if result['score'] < threshold_to_use else {
                'language': result['label'],
                'confidence': result['score'],
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