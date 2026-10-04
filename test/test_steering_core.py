import unittest

import torch
from torch import nn

from steering_recovery import (
    bound_l2,
    extract_seed_numbers,
    original_three_digit_sequence,
    project_l2_,
    register_shared_delta,
    remove_hooks,
    remove_seed_numbers,
    strict_three_digit_sequence,
)


class _Block(nn.Module):
    def forward(self, value):
        return (value + 1, "cache")


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block(), _Block()])


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()

    def forward(self, value):
        for layer in self.model.layers:
            value = layer(value)[0]
        return value


class SteeringVectorPipelineTests(unittest.TestCase):
    def test_l2_projection_is_hard_bound(self):
        parameter = nn.Parameter(torch.tensor([3.0, 4.0]))
        before, after = project_l2_(parameter, 2.0)
        self.assertAlmostEqual(before, 5.0)
        self.assertAlmostEqual(after, 2.0)

    def test_graph_bound_limits_norm_and_keeps_gradient(self):
        raw = nn.Parameter(torch.tensor([3.0, 4.0]))
        bounded = bound_l2(raw, 2.0)
        self.assertAlmostEqual(bounded.norm().item(), 2.0)
        bounded[0].backward()
        self.assertIsNotNone(raw.grad)
        self.assertGreater(raw.grad.abs().sum().item(), 0.0)

    def test_same_delta_is_shared_by_every_layer(self):
        model = _FakeModel()
        delta = nn.Parameter(torch.tensor([2.0]))
        hooks = register_shared_delta(model, delta)
        try:
            output = model(torch.tensor([0.0]))
            output.backward()
        finally:
            remove_hooks(hooks)
        self.assertEqual(len(hooks), 3)
        self.assertEqual(output.item(), 9.0)
        self.assertEqual(delta.grad.item(), 3.0)

    def test_zero_scaled_delta_is_unsteered(self):
        model = _FakeModel()
        delta = torch.tensor([2.0])
        hooks = register_shared_delta(model, delta * 0.0)
        try:
            output = model(torch.tensor([0.0]))
        finally:
            remove_hooks(hooks)
        self.assertEqual(output.item(), 3.0)

    def test_numeric_filter_requires_entire_string_and_one_separator(self):
        valid = "123, 456, 789, 101, 202, 303, 404, 505, 606, 707"
        accepted, _, cleaned = strict_three_digit_sequence(valid, 10, 40)
        self.assertTrue(accepted)
        self.assertEqual(cleaned, valid)
        invalid = (
            "Here are numbers: " + valid,
            "123, 456; 789, 101, 202, 303, 404, 505, 606, 707",
            "123, 456, 78, 101, 202, 303, 404, 505, 606, 707",
            "(123, 456, 789, 101, 202, 303, 404, 505, 606, 707]",
        )
        for text in invalid:
            self.assertFalse(strict_three_digit_sequence(text, 10, 40)[0])

    def test_original_filter_extracts_numbers_from_surrounding_text(self):
        text = "Here are numbers:\n123, 456, 789, 234, 567, 89"
        accepted, _, cleaned = original_three_digit_sequence(text, 5, 40)
        self.assertTrue(accepted)
        self.assertEqual(cleaned, "123, 456, 789, 234, 567")

    def test_original_filter_removes_prompt_seed_numbers(self):
        prompt = "Start with these numbers: 123, 456, 789. Generate numbers."
        completion = "123, 234, 456, 567, 789, 890, 345, 678"
        seed_numbers = extract_seed_numbers(prompt)
        cleaned = remove_seed_numbers(completion, seed_numbers)
        self.assertEqual(seed_numbers, {123, 456, 789})
        self.assertEqual(cleaned, "234, 567, 890, 345, 678")
        self.assertTrue(original_three_digit_sequence(cleaned, 5, 40)[0])


if __name__ == "__main__":
    unittest.main()
