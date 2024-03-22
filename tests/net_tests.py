import unittest
import torch
from torch import optim
import torch.nn.functional as F

from policy_value_networks import PolicyNet, ValueNet
import config


class TestPolicyNet(unittest.TestCase):
    def setUp(self):
        self.policy_net = PolicyNet()
        self.input_data = torch.randn(1, 1, 8, 8)  # Example input data

    def test_forward_pass(self):
        output = self.policy_net(self.input_data)
        self.assertEqual(list(output.shape), [1, 64])  # Check output shape

    def test_softmax_by_legal_moves(self):
        input_tensor = torch.randn(64, device=config.device)  # Example output tensor
        outputs = self.policy_net(input_tensor)
        mask = self.policy_net.create_mask(self.policy_net.action_labels[7:57])
        masked_outputs = outputs * mask.float()
        self.assertAlmostEqual(masked_outputs.sum().item(), 1.0 - 0.16,  delta=0.1)  # Check if probabilities sum by

    def test_get_one_hot(self):
        result = self.policy_net.get_one_hot_target((1, 1))
        correct_tensor = torch.zeros(64, device=config.device)
        correct_tensor[9] = 1.

        self.assertTrue(torch.allclose(result, correct_tensor))

    def test_update(self):
        input_tensor = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.,
                                     0., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0.,
                                     9., 10., 11., 12., 13., 14., 15., 16.], requires_grad=True,
                                    device=config.device)

        output_tensor = self.policy_net(input_tensor)
        target = (1, 1)
        reward = 1.

        self.policy_net.update(output_tensor, target, reward)

        for parm in self.policy_net.parameters():
            self.assertNotEqual(parm.grad, None)


class TestValueNet(unittest.TestCase):
    def setUp(self):
        self.value_net = ValueNet()
        self.input_data = torch.randn(1, 1, 8, 8)  # Example input data

    def test_forward_pass(self):
        output = self.value_net(self.input_data)
        self.assertEqual(list(output.shape), [1, 1])  # Check output shape

    def test_update(self):
        input_tensor = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.,
                                          0., 0., 0., 0., 0., 0., 0., 0.,
                                          0., 0., 0., 0., 0., 0., 0., 0.,
                                          0., 0., 0., 0., 0., 0., 0., 0.,
                                          0., 0., 0., 0., 0., 0., 0., 0.,
                                          0., 0., 0., 0., 0., 0., 0., 0.,
                                          0., 0., 0., 0., 0., 0., 0., 0.,
                                          9., 10., 11., 12., 13., 14., 15., 16.], requires_grad=True,
                                          device=config.device)

        output_tensor = self.value_net(input_tensor)
        target = 1.0  # Example target value
        self.value_net.update(output_tensor, target)  # Update network parameters
        # check gradient is not None
        for parm in self.value_net.parameters():
            self.assertNotEqual(parm.grad, None)


if __name__ == '__main__':
    unittest.main()
