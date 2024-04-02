import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Any, Dict, List, Optional
from torch import Tensor


@torch.jit.script
def compute_gating(k: int, num_experts: int, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    """
    Compute gating values for the mixture of experts based on probabilities and top-k indices.

    Args:
        k (int): Number of experts to select.
        num_experts (int): Total number of experts.
        top_k_gates (torch.Tensor): Gating values for top-k experts (batch_size x k).
        top_k_indices (torch.Tensor): Indices of top-k experts (batch_size x k).

    Returns:
        torch.Tensor: Batch-level gating values.
        torch.Tensor: Batch-level expert indices.
        torch.Tensor: Expert size for each expert.
        torch.Tensor: Sorted indices of top-k experts.
    """
    zeros = torch.zeros([top_k_gates.size(0), num_experts], dtype=top_k_gates.dtype, device=top_k_gates.device)
    gates = zeros.scatter(1, top_k_indices, 1)
    expert_size = gates.long().sum(0)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    _, index_sorted_experts = top_k_experts.sort(0)
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, index_sorted_experts


class ParallelLinear(torch.autograd.Function):
    """
    A custom autograd function for Parallel Linear operation.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, expert_size_list, weight, bias=None):
        """
        Forward pass of the ParallelLinear operation.

        Args:
            ctx: Context object.
            input (Tensor): Input tensor.
            expert_size_list (List[int]): List of expert sizes.
            weight (Tensor): Weight tensor.
            bias (Optional[Tensor]): Bias tensor.

        Returns:
            Tensor: Output tensor.
        """
        # expert_size_list: List[int] = expert_size.tolist()
        output = ParallelLinear.forward_scriptable(input, expert_size_list, weight, bias)
        # assert torch.allclose(ParallelLinear._forward_scriptable(input, expert_size, weight, bias),  output)
        ctx.save_for_backward(input, weight, bias)
        ctx.expert_size_list = expert_size_list
        return output

    @staticmethod
    @torch.jit.script
    def forward_scriptable(input: Tensor, expert_size_list: List[int],
                           weight: Tensor, bias: Optional[Tensor]):
        """
        Scriptable forward pass of the ParallelLinear operation.

        Args:
            input (Tensor): Input tensor.
            expert_size_list (List[int]): List of expert sizes.
            weight (Tensor): Weight tensor.
            bias (Optional[Tensor]): Bias tensor.

        Returns:
            Tensor: Output tensor.
        """
        output_buf: Tensor = torch.empty((input.size(0), weight.size(2)),
                                         device=input.device, dtype=input.dtype)
        num_linears = weight.size(0)

        input_list = input.split(expert_size_list, dim=0)
        output_buf_list = output_buf.split(expert_size_list)

        for i in range(num_linears):
            torch.mm(input_list[i], weight[i], out=output_buf_list[i])

        if bias is not None:
            for i in range(num_linears):
                output_buf_list[i].add_(bias[i])

        output = output_buf
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        """
        Backward pass of the ParallelLinear operation.

        Args:
            ctx: Context object.
            grad_out (Tensor): Gradient of the output.

        Returns:
            Tuple of Tensors: Gradients with respect to input, weight, and bias.
        """
        input, weight, bias = ctx.saved_tensors
        expert_size_list = ctx.expert_size_list
        return ParallelLinear.backward_scriptable(
            grad_out, input, expert_size_list,
            weight, bias
        )

    @staticmethod
    @torch.jit.script
    def backward_scriptable(grad_out: Tensor,
                 input: Tensor, expert_size_list: List[int],
                 weight: Tensor, bias: Optional[Tensor]):
        """
        Scriptable backward pass of the ParallelLinear operation.

        Args:
            grad_out (Tensor): Gradient of the output.
            input (Tensor): Input tensor.
            expert_size_list (List[int]): List of expert sizes.
            weight (Tensor): Weight tensor.
            bias (Optional[Tensor]): Bias tensor.

        Returns:
            Tuple of Tensors: Gradients with respect to input, weight, and bias.
        """
        num_linears = weight.size(0)
        input_list = input.t().split(expert_size_list, dim=1)
        grad_list = grad_out.split(expert_size_list, dim=0)

        d_input_buf = torch.empty_like(input)
        d_input_buf_list = d_input_buf.split(expert_size_list, dim=0)
        d_weight_buf = torch.empty_like(weight)

        weight_t = weight.permute(0, 2, 1)

        for i in range(num_linears):
            torch.mm(grad_list[i], weight_t[i], out=d_input_buf_list[i])
            torch.mm(input_list[i], grad_list[i], out=d_weight_buf[i])

        d_input = d_input_buf
        d_weight = d_weight_buf

        if bias is not None:
            d_bias_buf = torch.empty_like(bias)
            for i in range(num_linears):
                torch.sum(grad_list[i], dim=0, keepdim=False, out=d_bias_buf[i])
            d_bias = d_bias_buf
        else:
            d_bias = None

        return d_input, None, d_weight, d_bias


class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
        """
        Initialize the ParallelExperts module.

        Args:
            num_experts (int): Number of experts.
            input_size (int): Size of the input.
            output_size (int): Size of the output.
            bias (bool): Whether to include bias terms.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_experts, output_size))
        else:
            self.bias = None
        self.reset_parameters()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def extra_repr(self):
        return 'num_experts={}, input_size={}, output_size={}'.format(
            self.num_experts, self.input_size, self.output_size)

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the model.
        """
        nn.init.uniform_(self.weight, -1. / self.weight.size(1), 1. / self.weight.size(1))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs, expert_size):
        """
        Forward pass of the ParallelExperts module.

        Args:
            inputs (Tensor): Input tensor.
            expert_size: Expert size information.

        Returns:
            Tensor: Output tensor.
        """
        results = ParallelLinear.apply(inputs, expert_size, self.weight.transpose(1, 2), self.bias)
        # expert_size_list: List[int] = expert_size.tolist()
        # input_list = inputs.split(expert_size_list, dim=0)
        # output_list = []
        # for i in range(self.num_experts):
        #     output_list.append(self.input_experts[i](input_list[i]))
        # results = torch.cat(output_list, dim=0)
        return results