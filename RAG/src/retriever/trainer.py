import os
from typing import Optional

import torch

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from retriever.modeling import EncoderModel


import logging
logger = logging.getLogger(__name__)


class TevatronTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        # Handle both old and new transformers versions
        processing_class = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
        if processing_class is not None:
            processing_class.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _save_checkpoint(self, model, trial=None, metrics=None):
        if self.is_world_process_zero():
            try:
                # Ensure the checkpoint directory exists and has proper permissions
                checkpoint_dir = self.args.output_dir
                if hasattr(self.args, 'run_name'):
                    # Get the current step number for checkpoint naming
                    step = self.state.global_step
                    checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{step}")
                
                # Create directory if it doesn't exist
                os.makedirs(checkpoint_dir, exist_ok=True)
                logger.info(f"Saving checkpoint to {checkpoint_dir}")
                
                # Remove metrics parameter - not supported in current transformers version
                super()._save_checkpoint(model, trial=trial)
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")
                # Try to save just the model without optimizer/scheduler state
                try:
                    logger.info("Attempting to save model only...")
                    self._save(checkpoint_dir)
                except Exception as e2:
                    logger.error(f"Failed to save model: {e2}")
                    raise e

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        query, passage = inputs
        output = model(query=query, passage=passage)
        loss = output.loss
        return (loss, output) if return_outputs else loss

    def training_step(self, *args, **kwargs):
        loss = super(TevatronTrainer, self).training_step(*args, **kwargs)
        return loss / self._dist_loss_scale_factor