import faiss
import numpy as np
from tqdm import tqdm
import torch
import os

import logging

logger = logging.getLogger(__name__)

class FaissFlatSearcher:
    def __init__(self, init_reps: np.ndarray):
        index = faiss.IndexFlatIP(init_reps.shape[1])
        # Actually add the initial representations to the index
        index.add(init_reps.astype('float32'))
        self.index = index
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")

    def move_index_to_gpu(self):
        """Move index to GPU if available, otherwise stay on CPU"""
        try:
            # Check if CUDA is disabled via environment variable
            if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
                logger.info("CUDA disabled via CUDA_VISIBLE_DEVICES, staying on CPU")
                return
                
            # Check if GPU functionality is available in FAISS
            if not hasattr(faiss, 'StandardGpuResources'):
                logger.info("GPU functionality not available in FAISS, staying on CPU")
                return
                
            # Check if CUDA is available
            if not torch.cuda.is_available():
                logger.info("CUDA not available, staying on CPU")
                return
                
            logger.info("Moving index to GPU(s)")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            logger.info("Successfully moved index to GPU")
            
        except Exception as e:
            logger.warning(f"Failed to move index to GPU: {e}")
            logger.info("Continuing with CPU-only search")
            # Continue with CPU index
        
    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps.astype('float32'))

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps.astype('float32'), k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool=False):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices