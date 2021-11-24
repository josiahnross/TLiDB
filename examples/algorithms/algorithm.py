import os
import sys
import torch
import torch.nn as nn

from utils import move_to, detach_and_clone
from optimizers import initialize_optimizer
from losses import initialize_loss

class Algorithm(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.device = config.device
        self.out_device = 'cpu'
        self.optimizer = initialize_optimizer(config, model)
        self.max_grad_norm = config.max_grad_norm
        self.model = model
    
    # def objective(self, results, metric):
    #     raise NotImplementedError

    def process_batch(self, batch):
        raise NotImplementedError

    def update(self, batch):
        """
        Process the batch, and update the model
        Args:
            - batch: a batch of data yielded by data loaders
        Output:
            - results (dict): information about the batch, such as:
                - y_pred: the predicted labels
                - y_true: the true labels
                - metadata: the metadata of the batch
                - loss: the loss of the batch
                - metrics: the metrics of the batch
        """
        assert self.is_training, "Cannot update() when not in training mode"
        
        results, objective = self.process_batch(batch)
        self._apply_gradients(objective)
        return self.sanitize_dict(results)

    def _apply_gradients(self, objective):
        """
        A helper function for update() that applies the gradients to the model
        """
        objective.backward()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.model.zero_grad()



    # def _update(self, results):
    #     """
    #     Computes the objective and updates the model
    #     """
    #     metric = initialize_loss(results['objective']['loss_name'])
    #     objective = self.objective(results, metric)
    #     results['objective']['loss_value'] = objective.item()

    #     # update the model
    #     objective.backward()
    #     if self.max_grad_norm:
    #         nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
    #     self.optimizer.step()
    #     self.model.zero_grad()


    def evaluate(self, batch):
        """
        Process the batch, and evaluate the model
        Args:
            - batch: a batch of data yielded by data loaders
        Output:
            - results (dict): information about the batch, such as:
                - y_pred: the predicted labels
                - y_true: the true labels
                - metadata: the metadata of the batch
                - loss: the loss of the batch
                - metrics: the metrics of the batch
        """
        assert not self.is_training, "Cannot evaluate() when in training mode"
        results, _ = self.process_batch(batch)
        return self.sanitize_dict(results)

    def train(self, mode=True):
        """
        Set the model to training mode
        """
        self.is_training = mode
        super().train(mode)

    def eval(self):
        """
        Set the model to evaluation mode
        """
        self.train(False)

    def sanitize_dict(self, in_dict, to_out_device=True):
        """
        Helper function that sanitizes dictionaries by:
            - moving to the specified output device
            - removing any gradient information
            - detaching and cloning the tensors
        Args:
            - in_dict (dictionary)
        Output:
            - out_dict (dictionary): sanitized version of in_dict
        """
        out_dict = detach_and_clone(in_dict)
        if to_out_device:
            out_dict = move_to(out_dict, self.out_device)
        return out_dict