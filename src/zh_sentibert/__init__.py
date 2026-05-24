"""zh-sentibert: Chinese sentiment analysis with fine-tuned BERT."""

from .config import Config
from .model import ImprovedModel, MultiTaskModel, AttentionModel
from .data_utils import ImprovedDataset, DataProcessor, create_data_loaders
from .trainer import Trainer, evaluate_model
from .inference import SentimentPredictor

__version__ = "0.1.0"
__all__ = [
    "Config",
    "ImprovedModel",
    "MultiTaskModel",
    "AttentionModel",
    "ImprovedDataset",
    "DataProcessor",
    "create_data_loaders",
    "Trainer",
    "evaluate_model",
    "SentimentPredictor",
]
