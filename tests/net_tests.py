import unittest
import torch
from torch import optim
import torch.nn.functional as F

from policy_value_networks import PolicyNet, ValueNet


class TestPolicyNet(unittest.TestCase):
    def setUp(self):
        self.policy_net = PolicyNet()
        self.input_data = torch.randn(1, 1, 8, 8)  # Example input data

    def test_forward_pass(self):
        output = self.policy_net.forward(self.input_data)
        self.assertEqual(list(output.shape), [1, 64])  # Check output shape

    def test_create_mask(self):
        valid_actions = [(0, 0), (1, 1), (2, 2), (3, 3)]  # Example list of valid actions
        mask = self.policy_net.create_mask(valid_actions)
        self.assertEqual(mask.sum(), len(valid_actions))  # Check if mask is correct size
        output_tensor = list([1., 0., 0., 0., 0., 0., 0., 0.,
                              0., 1., 0., 0., 0., 0., 0., 0.,
                              0., 0., 1., 0., 0., 0., 0., 0.,
                              0., 0., 0., 1., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.])

        self.assertEqual(list(mask), output_tensor)

    def test_softmax_by_legal_moves(self):
        outputs = torch.randn(64)  # Example output tensor
        mask = self.policy_net.create_mask(self.policy_net.action_labels[:33])
        masked_outputs = self.policy_net.softmax_by_legal_moves(outputs, mask)
        self.assertAlmostEqual(masked_outputs.sum().item(), 1.0,  delta=1e-6)  # Check if probabilities sum to 1

    def test_update(self):
        pass


class TestValueNet(unittest.TestCase):
    def setUp(self):
        self.value_net = ValueNet()
        self.input_data = torch.randn(1, 1, 8, 8)  # Example input data

    def test_forward_pass(self):
        output = self.value_net(self.input_data)
        self.assertEqual(list(output.shape), [1, 1])  # Check output shape

    def test_update(self):
        output = torch.randn(1, requires_grad=True)  # Example output from network
        target = torch.randn(1)  # Example target value
        self.value_net.update(output, target)  # Update network parameters



if __name__ == '__main__':
    unittest.main()
