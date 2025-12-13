import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, feature_dim=64):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(feature_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, local, global_):
        """
        local:  [B, 64]   — local feature vector
        global_: [B, 64]   — global feature vector
        return: [B, 1]     — score for each (local, global) pair
        """
        h = torch.cat([local, global_], dim=1)  # [B, 128]
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)  # [B, 1]


class DeepInfoMaxLoss(nn.Module):
    """
    Copied from https://github.com/DuaneNielsen/DeepInfomaxPytorch
    """
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        # global discriminator
        self.global_d = Discriminator()

        self.local_d = LocalDiscriminator()
        self.prior_d = PriorDiscriminator()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, 26, 26)

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma

        return LOCAL + GLOBAL + PRIOR


if __name__ == '__main__':
    # global_discriminator = GlobalDiscriminator()
    # global_discriminator(torch.randn([32, 64]), torch.randn([32, 23, 64]))
    # pytorch_total_params = sum(p.numel() for p in global_discriminator.parameters() if p.requires_grad)
    # print(pytorch_total_params)

    local_discriminator = LocalDiscriminator()
    local_discriminator(torch.randn([32, 64]), torch.randn([32, 64]))
    pytorch_total_params = sum(p.numel() for p in local_discriminator.parameters() if p.requires_grad)
    print(pytorch_total_params)
