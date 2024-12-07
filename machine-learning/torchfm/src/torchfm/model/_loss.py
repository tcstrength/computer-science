import torch
import torch.nn as nn
import torch.nn.functional as F

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_scores, neg_scores):
        # pos_scores: Predicted scores for positive items (shape: [batch_size])
        # neg_scores: Predicted scores for negative items (shape: [batch_size])
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss

class WARPLoss(nn.Module):
    def __init__(self):
        super(WARPLoss, self).__init__()

    def forward(self, pos_scores, neg_scores):
        # pos_scores: Predicted scores for positive items (shape: [batch_size])
        # neg_scores: Predicted scores for negative items (shape: [batch_size, num_negatives])
        loss = 0.0
        for i in range(neg_scores.size(1)):  # Iterate through negative items
            neg_score = neg_scores[:, i]  # Get score of the current negative item
            loss += torch.mean(torch.log(1 + torch.exp(neg_score - pos_scores)))

        return loss