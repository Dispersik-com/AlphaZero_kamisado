import unittest
import random
import torch
import config
from policy_value_networks import PolicyNet, ValueNet


class TestModels(unittest.TestCase):

    def setUp(self):
        self.policy_net = PolicyNet()
        self.value_net = ValueNet()
        self.input_tensor = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.,
                                     0., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0.,
                                     9., 10., 11., 12., 13., 14., 15., 16.], requires_grad=True,
                                    device=config.device)

    def test_policy_net_forward(self):
        input_data = torch.randn(1, 10, 64)
        print(input_data.size())
        output = self.policy_net(input_data)
        self.assertEqual(output.size(), (1, 64))
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))

    def test_value_net_forward(self):
        input_data = torch.randn(1, 10, 64)
        output = self.value_net(input_data)
        self.assertEqual(output.size(), (1, 1))
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))

    def test_policy_net_batch_update(self):
        time_len = 10
        outputs = []
        targets = []
        rewards = []
        input_tensors = []
        for _ in range(time_len):
            input_tensors.append(self.input_tensor.clone())
            time_seq = torch.stack(input_tensors).clone().view(1, len(input_tensors), 64)
            outputs.append(self.policy_net(time_seq))
            targets.append(random.choice(self.policy_net.action_labels))
            rewards.append(random.random())

        loss = self.policy_net.batch_update(outputs, targets, rewards)

        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss))

        for name, param in self.policy_net.named_parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.sum(param.grad) != 0.0, f"Gradient for parameter {name} is zero")

    def test_value_net_batch_update(self):
        time_len = 5
        outputs = []
        rewards = []
        input_tensors = []
        for _ in range(time_len):
            input_tensors.append(self.input_tensor.clone())
            time_seq = torch.stack(input_tensors).clone().view(1, len(input_tensors), 64)
            outputs.append(self.value_net(time_seq))
            rewards.append(random.random())

        loss = self.value_net.batch_update(outputs, rewards)

        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss))

        for name, param in self.value_net.named_parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.sum(param.grad) != 0.0, f"Gradient for parameter {name} is zero")


if __name__ == '__main__':
    unittest.main()
