import torch
import config


class SaveLoadInterface:

    def save_model(self, file_path):
        """
        Saves the model parameters to a file.

        Args:
            file_path (str): The path to the file where the model parameters will be saved.
        """
        torch.save(self.state_dict(), file_path)

    @classmethod
    def load_model(cls, file_path):
        """
        Creates a new instance of the model and loads parameters from a file.

        Args:
            file_path (str): The path to the file from which the model parameters will be loaded.

        Returns:
            PolicyNet: The loaded model.
        """
        model = cls()
        model.load_state_dict(torch.load(file_path, map_location=torch.device(config.device)))
        model.eval()  # Set the model to evaluation mode
        return model


# class NetRegularizationInterface:
#
#     def l1_regularization(model, l1_lambda):
#         l1_reg = torch.tensor(0., requires_grad=True)
#         for param in model.parameters():
#             l1_reg = l1_reg + torch.norm(param, 1)
#         return l1_lambda * l1_reg
#
#     def l2_regularization(model, l2_lambda):
#         l2_reg = torch.tensor(0., requires_grad=True)
#         for param in model.parameters():
#             l2_reg = l2_reg + torch.norm(param, 2)
#         return l2_lambda * l2_reg
