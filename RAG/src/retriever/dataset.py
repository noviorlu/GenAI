import random
from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

from retriever.arguments import DataArguments

import logging
logger = logging.getLogger(__name__)


def format_query(query: str, prefix: str = 'Query: ') -> str:
    return f'{prefix} {query.strip()}'.strip()

def format_passage(text: str, title: str = '', prefix: str = 'Passage: ') -> str:
    return f'{prefix} {title.strip()} {text.strip()}'.strip()


class TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        
        # Load the sentence-transformers dataset
        if "sentence-transformers/msmarco-hard-negatives" in self.data_args.dataset_name:
            self.train_data = load_dataset(
                self.data_args.dataset_name,
                split=self.data_args.dataset_split or 'train',
                cache_dir=self.data_args.dataset_cache_dir,
                streaming=True
            )
            self.use_sentence_transformers_format = True
            
            # Load the corpus to get actual text for passage IDs - FIXED with config
            self.corpus_data = load_dataset(
                "sentence-transformers/msmarco-corpus", 
                'passage',  # Added config parameter
                split='corpus',
                cache_dir=self.data_args.dataset_cache_dir,
                streaming=True
            )
            # Convert corpus to dictionary for fast lookup
            self.corpus_dict = {}
            logger.info("Loading corpus for passage lookup...")
            for i, item in enumerate(self.corpus_data):
                self.corpus_dict[item['docid']] = {
                    'text': item['text'], 
                    'title': item.get('title', '')
                }
                if i >= 50000:  # Reduced to 50k to save memory
                    break
            logger.info(f"Loaded {len(self.corpus_dict)} passages into corpus dict")
            
            # Load queries - FIXED with config
            self.queries_data = load_dataset(
                "sentence-transformers/msmarco-queries",
                'queries',  # Added config parameter  
                split='train',  # Changed split name
                cache_dir=self.data_args.dataset_cache_dir,
                streaming=True
            )
            # Convert queries to dictionary
            self.queries_dict = {}
            logger.info("Loading queries...")
            for i, item in enumerate(self.queries_data):
                self.queries_dict[item['qid']] = item['query']
                if i >= 50000:  # Reduced to 50k to save memory
                    break
            logger.info(f"Loaded {len(self.queries_dict)} queries into queries dict")
            
        elif "microsoft/ms_marco" in self.data_args.dataset_name:
            # Handle Microsoft MS MARCO format
            self.train_data = load_dataset(
                self.data_args.dataset_name,
                'v1.1',  # Add config for microsoft dataset
                split=self.data_args.dataset_split or 'train',
                cache_dir=self.data_args.dataset_cache_dir,
                streaming=True
            )
            self.use_sentence_transformers_format = False
            self.use_microsoft_format = True
        else:
            # Use standard Tevatron format for other datasets
            self.train_data = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                data_files=self.data_args.dataset_path,
                split=self.data_args.dataset_split,
                cache_dir=self.data_args.dataset_cache_dir,
                streaming=True
            )
            self.use_sentence_transformers_format = False
            self.use_microsoft_format = False
        
        if self.data_args.dataset_number_of_shards > 1:
            self.train_data = self.train_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

        # Convert to list and slice
        if self.use_sentence_transformers_format:
            logger.info("Converting streaming dataset to list...")
            self.train_data = list(self.train_data)  
        else:
            self.train_data = list(self.train_data)  

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        
        if self.use_sentence_transformers_format:
            # Handle sentence-transformers format
            qid = group['qid']
            query = self.queries_dict.get(qid, f"Query {qid}")
            
            # Get positive passages
            pos_ids = group['pos']
            group_positives = []
            for pid in pos_ids[:1]:  # Take only first positive
                if pid in self.corpus_dict:
                    group_positives.append(self.corpus_dict[pid])
            
            # Get negative passages (use BM25 negatives)
            neg_dict = group['neg']
            bm25_negs = neg_dict.get('bm25', [])
            group_negatives = []
            for pid in bm25_negs[:20]:  # Take first 20 negatives
                if pid in self.corpus_dict:
                    group_negatives.append(self.corpus_dict[pid])
                    
        elif hasattr(self, 'use_microsoft_format') and self.use_microsoft_format:
            # Handle Microsoft MS MARCO format
            query = group['query']
            
            # Debug: Print the structure of the group to understand the data format
            if item == 0:  # Only print for first item to avoid spam
                logger.info(f"Sample group keys: {list(group.keys())}")
                logger.info(f"Sample group structure: {group}")
            
            # Handle different possible structures for passages
            passages_data = group.get('passages', group.get('passage', {}))
            
            # Initialize variables
            group_positives = []
            group_negatives = []
            
            # Handle case where passages is a dictionary with passage texts
            if isinstance(passages_data, dict):
                # Check for different possible keys in the passages dictionary
                passage_texts = []
                
                # Look for various possible field names
                for key in ['text', 'passage_text', 'passage', 'content']:
                    if key in passages_data:
                        if isinstance(passages_data[key], list):
                            passage_texts.extend(passages_data[key])
                        else:
                            passage_texts.append(passages_data[key])
                
                # If no text found in expected keys, try iterating through all values
                if not passage_texts:
                    for key, value in passages_data.items():
                        if isinstance(value, str) and len(value.strip()) > 0:
                            passage_texts.append(value)
                        elif isinstance(value, list):
                            passage_texts.extend([v for v in value if isinstance(v, str) and len(v.strip()) > 0])
                
            # Handle case where passages is a list
            elif isinstance(passages_data, list):
                passage_texts = [p for p in passages_data if isinstance(p, str)]
            
            # Handle case where passages_data is a string
            elif isinstance(passages_data, str):
                passage_texts = [passages_data]
            
            else:
                # Fallback: look for other possible fields that might contain passages
                passage_texts = []
                for key in ['text', 'passage', 'document', 'context']:
                    if key in group:
                        if isinstance(group[key], str):
                            passage_texts.append(group[key])
                        elif isinstance(group[key], list):
                            passage_texts.extend([p for p in group[key] if isinstance(p, str)])
            
            # Get answers for determining positive passages
            answers = group.get('wellFormedAnswers', group.get('answers', []))
            if not answers:
                # Try other possible answer field names
                answers = group.get('answer', group.get('response', []))
            
            # Ensure answers is a list
            if isinstance(answers, str):
                answers = [answers]
            elif not isinstance(answers, list):
                answers = []
            
            # Process passages to identify positives and negatives
            if passage_texts:
                for passage_text in passage_texts:
                    if not isinstance(passage_text, str) or len(passage_text.strip()) == 0:
                        continue
                        
                    is_positive = False
                    
                    # Check if any answer appears in this passage
                    for answer in answers:
                        if isinstance(answer, str) and len(answer.strip()) > 0:
                            if answer.lower().strip() in passage_text.lower():
                                is_positive = True
                                break
                    
                    passage_data = {
                        'text': passage_text.strip(),
                        'title': ''  # Microsoft format typically has no titles
                    }
                    
                    if is_positive and not group_positives:
                        group_positives.append(passage_data)
                    else:
                        group_negatives.append(passage_data)
                
                # If no positive found, treat first passage as positive
                if not group_positives and passage_texts:
                    group_positives.append({
                        'text': passage_texts[0].strip(),
                        'title': ''
                    })
                    group_negatives = [{'text': p.strip(), 'title': ''} for p in passage_texts[1:] if isinstance(p, str) and len(p.strip()) > 0]
            
            # If still no passages found, create a dummy positive passage
            if not group_positives and not group_negatives:
                logger.warning(f"No passages found for item {item}, creating dummy passage")
                group_positives.append({
                    'text': f"Dummy passage for query: {query}",
                    'title': ''
                })
                
        else:
            # Handle standard Tevatron format
            query = group['query']
            group_positives = group['positive_passages']
            group_negatives = group['negative_passages']

        formated_query = format_query(query)
        formated_passages = []

        # Add positive passage
        if group_positives:
            pos_psg = group_positives[0]
            formated_passages.append(format_passage(pos_psg['text'], pos_psg.get('title', '')))

        # Add negative passages
        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size) if group_negatives else []
        elif self.data_args.train_group_size == 1:
            negs = []
        else:
            negs = group_negatives[:negative_size]

        for neg_psg in negs:
            formated_passages.append(format_passage(neg_psg['text'], neg_psg.get('title', '')))

        return formated_query, formated_passages


class EncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        
        # Handle local JSONL files for BeIR datasets
        if self.data_args.dataset_path and self.data_args.dataset_path.endswith('.jsonl'):
            import json
            self.encode_data = []
            with open(self.data_args.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        self.encode_data.append(json.loads(line))
        else:
            self.encode_data = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                data_files=self.data_args.dataset_path,
                split=self.data_args.dataset_split,
                cache_dir=self.data_args.dataset_cache_dir,
                # Removed trust_remote_code=True - no longer supported
            )
            if self.data_args.dataset_number_of_shards > 1:
                self.encode_data = self.encode_data.shard(
                    num_shards=self.data_args.dataset_number_of_shards,
                    index=self.data_args.dataset_shard_index,
                )

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, str]:
        text = self.encode_data[item]
        
        # Handle BeIR format with _id and text fields
        if isinstance(text, dict) and '_id' in text and 'text' in text:
            text_id = text['_id']
            if self.data_args.encode_is_query:
                formated_text = format_query(text['text'])
            else:
                formated_text = format_passage(text['text'], text.get('title', ''))
        # Fallback to original format
        elif self.data_args.encode_is_query:
            text_id = text.get('query_id', text.get('_id', str(item)))
            formated_text = format_query(text.get('query', text.get('text', '')))
        else:
            text_id = text.get('docid', text.get('_id', str(item)))
            formated_text = format_passage(
                text.get('text', ''), 
                text.get('title', '')
            )
        
        return text_id, formated_text