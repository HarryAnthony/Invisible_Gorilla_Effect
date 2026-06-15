from typing import List

import torch


class ChannelConcept:
    """Concept class for torch.nn.Conv2d and torch.nn.Linear layers."""

    @staticmethod
    def mask(batch_id: int, concept_ids: List, layer_name=None):
        def mask_fct(grad):
            mask = torch.zeros_like(grad[batch_id])
            mask[concept_ids] = 1
            grad[batch_id] = grad[batch_id] * mask
            return grad

        return mask_fct

    def attribute(self, relevance, mask=None, layer_name: str = None, abs_norm=True):
        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask

        rel_l = torch.sum(relevance.view(*relevance.shape[:2], -1), dim=-1)

        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1).view(-1, 1) + 1e-10)

        return rel_l
