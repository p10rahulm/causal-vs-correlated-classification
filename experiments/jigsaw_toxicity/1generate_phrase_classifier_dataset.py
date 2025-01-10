import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
from utilities.general_utilities import get_claude_response, save_results, set_seed, append_to_json
import json
import textwrap
from data_loaders.imdb import get_imdb_train_samples
from utilities.phrase_extraction import remove_punctuation_phrases, extract_phrases
import pandas as pd
import re
import argparse
from tqdm import tqdm 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



@dataclass
class DatasetConfig:
    """Configuration for each dataset."""
    name: str
    classification_word: str
    file_path: Optional[str]
    input_column: str
    label_column: str
    sample_size: int

DATASET_CONFIGS = {
    'imdb': DatasetConfig(
        name='imdb',
        classification_word='sentiment',
        file_path=None,
        input_column='text',
        label_column='label',
        sample_size=1000
    ),
    'jailbreak': DatasetConfig(
        name='jailbreak',
        classification_word='toxic',
        file_path='toxic-chat_annotation_train.csv',
        input_column='user_input',
        label_column='toxicity',
        sample_size=1000
    ),
    'jigsaw_toxicity': DatasetConfig(
        name='jigsaw_toxicity',
        classification_word='toxic',
        file_path=Path('../../data/toxicity_data/train.csv'),
        input_column='comment_text',
        label_column='toxic',
        sample_size=1000
    ),
    'olid': DatasetConfig(
        name='olid',
        classification_word='offensive',
        file_path=Path('data/olid_data/olid-training-v1.0.tsv'),
        input_column='tweet',
        label_column='subtask_a',
        sample_size=1000
    )
}


def remove_non_ascii(text: str) -> str:
    """Remove non-ASCII characters from text."""
    if isinstance(text, str):
        return re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def clean_text(text: str) -> str:
    """Clean text by removing HTML tags, quotes, and normalizing whitespace."""
    if not isinstance(text, str):
        return text
    
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.replace('"', "'")  # Replace double quotes with single quotes
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    return text

def read_examples_from_file(classification_word: str) -> str:
    """Read example templates from file with proper error handling."""
    template_dir = Path("../../prompt_templates/wz_classification")
    file_path = template_dir / f"{classification_word.lower()}.txt"
    
    try:
        return file_path.read_text().strip()
    except FileNotFoundError:
        logger.warning(f"Example file for {classification_word} not found. Using default sentiment examples.")
        default_path = template_dir / "sentiment.txt"
        return default_path.read_text().strip()
    except Exception as e:
        logger.error(f"Error reading template file: {e}")
        raise


def prompt_builder(
    phrases: List[str], 
    full_text: str, 
    classification_word: str = "sentiment", 
    dataset: str = 'imdb'
) -> str:
    """Build prompt for phrase classification."""
    phrases_str = ", ".join(f'"{phrase}"' for phrase in phrases)
    examples = read_examples_from_file(classification_word)
    
    suffix = " attribute" if dataset != 'imdb' else ""
    
    return textwrap.dedent(f"""
        You are given the following full text:
        "{full_text}"
        From this text, the following phrases have been extracted:
        {phrases_str}
        Classify each phrase into one of two categories:
        "{classification_word}_phrases": Those phrases that are directly related to or express {classification_word.lower()}{suffix}.
        "neutral_phrases": Those phrases that are not directly related to {classification_word.lower()}{suffix}.
        {examples}
        Now, classify the extracted phrases from the given text based on the classification word "{classification_word}":
        Output your classification as a JSON object with two keys: "{classification_word.lower()}_phrases" and "neutral_phrases", each containing a list of the corresponding phrases.
        IMPORTANT: Your response must be ONLY valid JSON that matches the structure described. Do not include any explanatory text before or after the JSON. Begin your response with the opening curly brace '{{' and end with the closing curly brace '}}'.
    """).strip()


def classify_phrases(
    phrases: List[str], 
    full_text: str, 
    classification_word: str = "sentiment", 
    dataset: str = 'imdb'
) -> Dict[str, List[str]]:
    """Classify phrases using Claude API."""
    user_prompt = prompt_builder(phrases, full_text, classification_word, dataset)
    return get_claude_response(user_prompt)



def process_texts(
    texts: List[str],
    labels: List[Any],
    classification_word: str = "sentiment",
    num_samples: Optional[int] = None,
    output_file: Optional[Path] = None,
    dataset: str = 'imdb'
) -> None:
    """Process and classify phrases from texts."""
    for i, (text, label) in tqdm(enumerate(zip(texts, labels)), desc="Processing texts"):
        if num_samples is not None and i >= num_samples:
            break
            
        try:
            clean_text_str = clean_text(text)
            extracted_phrases = remove_punctuation_phrases(extract_phrases(clean_text_str))
            
            if not extracted_phrases:
                logger.warning(f"No phrases extracted from text {i}")
                continue
                
            analysis = classify_phrases(extracted_phrases, clean_text_str, classification_word, dataset)
            if analysis:
                result = {
                    'text': clean_text_str,
                    f'{classification_word.lower()}_phrases': analysis.get(f'{classification_word.lower()}_phrases', []),
                    'neutral_phrases': analysis.get('neutral_phrases', []),
                    'label': label
                }
                
                if output_file:
                    append_to_json(result, output_file)
            else:
                logger.warning(f"Failed to get valid classification for text {i}")
                
        except Exception as e:
            logger.error(f"Error processing text {i}: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Script for phrase classification")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        default='imdb',
        help='Dataset to use for classification'
    )
    parser.add_argument(
        '--ood_mode',
        type=str,
        default='sentiment',
        help='Out-of-distribution mode for classification'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Number of samples to process (overrides default sample size)'
    )
    return parser.parse_args()



def load_dataset_samples(config: DatasetConfig, random_seed: int = 42) -> Tuple[List[str], List[Any]]:
    """Load and prepare dataset samples based on configuration."""
    if config.name == 'imdb':
        return get_imdb_train_samples(n=config.sample_size)
        
    if config.name == 'jailbreak':
        dataset = load_dataset("lmsys/toxic-chat", split='train')
        df = pd.DataFrame(dataset)
    else:
        df = pd.read_csv(config.file_path, sep='\t' if config.name == 'olid' else ',')
    
    # Sample data
    df = df.sample(n=config.sample_size, random_state=random_seed)
    
    # Handle OLID specific preprocessing
    if config.name == 'olid':
        le = LabelEncoder()
        df[config.label_column] = le.fit_transform(df[config.label_column])
    
    # Clean the text data
    df[config.input_column] = df[config.input_column].apply(clean_text)
    
    return df[config.input_column].tolist(), df[config.label_column].tolist()

def main() -> None:
    """Main execution function."""
    args = parse_args()
    set_seed(42)

    # Get dataset configuration
    config = DATASET_CONFIGS[args.dataset]
    if args.sample_size:
        config.sample_size = args.sample_size
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('../../outputs') / f"{config.name}_phrase_dataset"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        raise
    
    # Load and process dataset
    try:
        texts, labels = load_dataset_samples(config)
        
        output_file = output_dir / f"{config.name}_{config.classification_word.lower()}_phrases_{timestamp}.json"
        process_texts(
            texts,
            labels,
            config.classification_word,
            output_file=output_file,
            dataset=config.name
        )
        logger.info(
            f"Processed {len(texts)} texts and saved results to {output_file}"
        )
        
    except Exception as e:
        logger.error(f"Error processing dataset {config.name}: {e}")
        raise

if __name__ == "__main__":
    main()
