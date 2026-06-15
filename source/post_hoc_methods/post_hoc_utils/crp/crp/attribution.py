"""
Conditional Concept Relevance Propagation utilities used by the PCX OOD method.

Adapted from zennit-crp (https://github.com/rachtibat/zennit-crp), built on zennit
(https://github.com/chr5tphr/zennit). See `post_hoc_utils/crp/ReadMe` for attribution
and citation details.
"""
from collections import namedtuple
from typing import Callable, Dict, List, Union

import numpy as np
import torch
import warnings

from source.post_hoc_methods.post_hoc_utils.zennit.src.zennit.composites import NameMapComposite
from source.post_hoc_methods.post_hoc_utils.zennit.src.zennit.core import Composite
from source.post_hoc_methods.post_hoc_utils.crp.crp.hooks import MaskHook
from source.post_hoc_methods.post_hoc_utils.crp.crp.concepts import ChannelConcept

attrResult = namedtuple("AttributionResults", "heatmap, activations, relevances, prediction")


class CondAttribution:

    def __init__(self, model: torch.nn.Module, device: torch.device = None, overwrite_data_grad=True, no_param_grad=True) -> None:
        self.MODEL_OUTPUT_NAME = "y"
        self.device = next(model.parameters()).device if device is None else device
        self.model = model
        self.overwrite_data_grad = overwrite_data_grad

        if no_param_grad:
            self.model.requires_grad_(False)

    def backward(self, pred, grad_mask, partial_backward, layer_names, layer_out, generate=False):
        if partial_backward and len(layer_names) > 0:
            wrt_tensor, grad_tensors = pred, grad_mask.to(pred)

            for l_name in layer_names:
                inputs = layer_out[l_name]
                try:
                    grad = torch.autograd.grad(
                        wrt_tensor, inputs=inputs, grad_outputs=grad_tensors, retain_graph=True
                    )
                except RuntimeError as e:
                    if "allow_unused=True" not in str(e):
                        raise e
                    raise RuntimeError(
                        "The layer names must be ordered according to their succession in the model if "
                        "'exclude_parallel'=True. Please make sure to start with the last and end with the first "
                        "layer in each condition dict. In addition, parallel layers can not be used in one condition."
                    )

                if grad is None:
                    raise RuntimeError(
                        "The layer names must be ordered according to their succession in the model if "
                        "'exclude_parallel'=True. Please make sure to start with the last and end with the first "
                        "layer in each condition dict. In addition, parallel layers can not be used in one condition."
                    )

                wrt_tensor, grad_tensors = layer_out[l_name], grad

            torch.autograd.backward(wrt_tensor, grad_tensors, retain_graph=generate)
        else:
            torch.autograd.backward(pred, grad_mask.to(pred), retain_graph=generate)

    def relevance_init(self, prediction, target_list, init_rel):
        if callable(init_rel):
            output_selection = init_rel(prediction)
        elif isinstance(init_rel, torch.Tensor):
            output_selection = init_rel
        elif isinstance(init_rel, (int, np.integer)):
            output_selection = torch.full(prediction.shape, init_rel)
        else:
            output_selection = prediction

        if target_list:
            mask = torch.zeros_like(output_selection)
            for i, targets in enumerate(target_list):
                mask[i, targets] = output_selection[i, targets]
            output_selection = mask

        return output_selection

    def heatmap_modifier(self, data, on_device=None):
        heatmap = data.grad.detach()
        heatmap = heatmap.to(on_device) if on_device else heatmap
        return torch.sum(heatmap, dim=1)

    def broadcast(self, data, conditions):
        len_data, len_cond = len(data), len(conditions)

        if len_data == len_cond:
            data.retain_grad()
            return data, conditions

        if len_cond > 1:
            data = torch.repeat_interleave(data, len_cond, dim=0)
        if len_data > 1:
            conditions = conditions * len_data

        data.retain_grad()
        return data, conditions

    def _check_arguments(self, data, conditions, start_layer, exclude_parallel, init_rel):
        if not data.requires_grad:
            raise ValueError("requires_grad attribute of 'data' must be True.")

        if self.overwrite_data_grad:
            data.grad = None
        elif data.grad is not None:
            warnings.warn(
                "'data' already has a filled .grad attribute. Set to None if not intended or set "
                "'overwrite_grad' to True."
            )

        distinct_cond = set()
        for cond in conditions:
            if self.MODEL_OUTPUT_NAME not in cond and start_layer is None and init_rel is None:
                raise ValueError(
                    f"Either {self.MODEL_OUTPUT_NAME} in 'conditions' or 'start_layer' or 'init_rel' must be defined."
                )

            if self.MODEL_OUTPUT_NAME in cond and start_layer is not None:
                warnings.warn(
                    f"You defined a condition for {self.MODEL_OUTPUT_NAME} that has no effect, since the "
                    f"'start_layer' {start_layer} is provided where the backward pass begins. If this behaviour "
                    "is not wished, remove 'start_layer'."
                )

            if exclude_parallel:
                if len(distinct_cond) == 0:
                    distinct_cond.update(cond.keys())
                elif distinct_cond ^ set(cond.keys()):
                    raise ValueError(
                        "If the 'exclude_parallel' flag is set to True, each condition dict must contain the "
                        "same layer names. (This limitation does not apply to the __call__ method)"
                    )

    def _register_mask_fn(self, hook, mask_map, b_index, c_indices, l_name):
        if callable(mask_map):
            mask_fn = mask_map(b_index, c_indices, l_name)
        elif isinstance(mask_map, Dict):
            mask_fn = mask_map[l_name](b_index, c_indices, l_name)
        else:
            raise ValueError("<mask_map> must be a dictionary or callable function.")

        hook.fn_list.append(mask_fn)

    def __call__(
        self, data: torch.tensor, conditions: List[Dict[str, List]],
        composite: Composite = None, record_layer: List[str] = [],
        mask_map: Union[Callable, Dict[str, Callable]] = ChannelConcept.mask, start_layer: str = None, init_rel=None,
        on_device: str = None, exclude_parallel=True) -> attrResult:
        if exclude_parallel:
            return self._conditions_wrapper(
                data, conditions, composite, record_layer, mask_map, start_layer, init_rel, on_device, True
            )
        return self._attribute(
            data, conditions, composite, record_layer, mask_map, start_layer, init_rel, on_device, False
        )

    def _conditions_wrapper(self, *args):
        data, conditions = args[:2]
        relevances, activations = {}, {}
        heatmap, prediction = None, None
        dist_conds = self._separate_conditions(conditions)

        for dist_layer in dist_conds:
            attr = self._attribute(data, dist_conds[dist_layer], *args[2:])

            for l_name in attr.relevances:
                if l_name not in relevances:
                    relevances[l_name] = attr.relevances[l_name]
                    activations[l_name] = attr.activations[l_name]
                else:
                    relevances[l_name] = torch.cat([relevances[l_name], attr.relevances[l_name]], dim=0)
                    activations[l_name] = torch.cat([activations[l_name], attr.activations[l_name]], dim=0)

            if heatmap is None:
                heatmap = attr.heatmap
                prediction = attr.prediction
            else:
                heatmap = torch.cat([heatmap, attr.heatmap], dim=0)
                prediction = torch.cat([prediction, attr.prediction], dim=0)

        return attrResult(heatmap, activations, relevances, prediction)

    def _separate_conditions(self, conditions):
        distinct_cond = dict()
        for cond in conditions:
            cond_set = frozenset(cond.keys())
            if cond_set in distinct_cond:
                distinct_cond[cond_set].append(cond)
            else:
                distinct_cond[cond_set] = [cond]
        return distinct_cond

    def _register_input_hook(self, data):
        def hook(grad):
            data.grad = grad
        return data.register_hook(hook)

    def _attribute(
        self, data: torch.tensor, conditions: List[Dict[str, List]],
        composite: Composite = None, record_layer: List[str] = [],
        mask_map: Union[Callable, Dict[str, Callable]] = ChannelConcept.mask, start_layer: str = None, init_rel=None,
        on_device: str = None, exclude_parallel=True) -> attrResult:
        data, conditions = self.broadcast(data, conditions)
        self._check_arguments(data, conditions, start_layer, exclude_parallel, init_rel)

        hook_map, y_targets, cond_l_names = {}, [], []
        for i, cond in enumerate(conditions):
            for l_name, indices in cond.items():
                if l_name == self.MODEL_OUTPUT_NAME:
                    y_targets.append(indices)
                else:
                    if l_name not in hook_map:
                        hook_map[l_name] = MaskHook([])
                    self._register_mask_fn(hook_map[l_name], mask_map, i, indices, l_name)
                    if l_name not in cond_l_names:
                        cond_l_names.append(l_name)

        handles, layer_out = self._append_recording_layer_hooks(record_layer, start_layer, cond_l_names)
        input_handle = self._register_input_hook(data)

        name_map = [([name], hook) for name, hook in hook_map.items()]
        mask_composite = NameMapComposite(name_map)

        if composite is None:
            composite = Composite()

        with mask_composite.context(self.model), composite.context(self.model) as modified:
            if start_layer:
                _ = modified(data)
                pred = layer_out[start_layer]
                grad_mask = self.relevance_init(pred.detach().clone(), None, init_rel)
                if start_layer in cond_l_names:
                    cond_l_names.remove(start_layer)
                self.backward(pred, grad_mask, exclude_parallel, cond_l_names, layer_out)
            else:
                pred = modified(data)
                grad_mask = self.relevance_init(pred.detach().clone(), y_targets, init_rel)
                self.backward(pred, grad_mask, exclude_parallel, cond_l_names, layer_out)

            attribution = self.heatmap_modifier(data, on_device)
            activations, relevances = {}, {}
            if len(layer_out) > 0:
                activations, relevances = self._collect_hook_activation_relevance(layer_out, on_device)

            activations['input'] = data.detach()
            relevances['input'] = data.grad.detach() if data.grad is not None else None

            [h.remove() for h in handles]
            input_handle.remove()

        return attrResult(attribution, activations, relevances, pred)

    @staticmethod
    def _generate_hook(layer_name, layer_out):
        def get_tensor_hook(module, input, output):
            layer_out[layer_name] = output
            output.retain_grad()

        return get_tensor_hook

    def _append_recording_layer_hooks(self, record_l_names: list, start_layer, cond_l_names):
        handles = []
        layer_out = {}
        record_l_names = record_l_names.copy()

        for l_name in cond_l_names:
            if l_name not in record_l_names:
                record_l_names.append(l_name)

        if start_layer is not None and start_layer not in record_l_names:
            record_l_names.append(start_layer)

        for name, layer in self.model.named_modules():
            if name == self.MODEL_OUTPUT_NAME:
                raise ValueError(
                    "No layer name should match the constant for the identifier of the model output."
                    "Please change the layer name or the OUTPUT_NAME constant of the object."
                    "Note, that the condition set then references to the output with OUTPUT_NAME and no longer 'y'."
                )

            if name in record_l_names:
                h = layer.register_forward_hook(self._generate_hook(name, layer_out))
                handles.append(h)
                record_l_names.remove(name)

        if start_layer in record_l_names:
            raise KeyError(f"<start_layer> {start_layer} not found in model.")
        if len(record_l_names) > 0:
            warnings.warn(f"Some layer names not found in model: {record_l_names}.")

        return handles, layer_out

    def _collect_hook_activation_relevance(self, layer_out, on_device=None, length=None):
        relevances = {}
        activations = {}
        for name in layer_out:
            act = layer_out[name].detach()[:length]
            activations[name] = act.to(on_device) if on_device else act
            activations[name].requires_grad = False

            if layer_out[name].grad is None:
                rel = torch.zeros_like(activations[name], requires_grad=False)[:length]
                relevances[name] = rel.to(on_device) if on_device else rel
            else:
                rel = layer_out[name].grad.detach()[:length]
                relevances[name] = rel.to(on_device) if on_device else rel
                relevances[name].requires_grad = False
                layer_out[name].grad = None

        return activations, relevances
