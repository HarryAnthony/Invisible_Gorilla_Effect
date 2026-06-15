import weakref
import functools

from source.post_hoc_methods.post_hoc_utils.zennit.src.zennit.core import RemovableHandle, RemovableHandleList


class MaskHook:
    '''Mask hooks for adaptive gradient masking or simple modification.'''

    def __init__(self, fn_list):
        self.fn_list = fn_list

    def post_forward(self, module, input, output):
        hook_ref = weakref.ref(self)

        @functools.wraps(self.backward)
        def wrapper(grad):
            return hook_ref().backward(module, grad)

        if not isinstance(output, tuple):
            output = (output,)

        if output[0].grad_fn is not None:
            output[0].register_hook(wrapper)
        return output[0] if len(output) == 1 else output

    def backward(self, module, grad):
        for mask_fn in self.fn_list:
            grad = mask_fn(grad)
        return grad

    def copy(self):
        return self.__class__(fn_list=self.fn_list)

    def remove(self):
        self.fn_list.clear()

    def register(self, module):
        return RemovableHandleList([
            RemovableHandle(self),
            module.register_forward_hook(self.post_forward),
        ])
