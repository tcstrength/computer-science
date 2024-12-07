import numpy as np

def auc_score(predicted_scores, many_hot_vector):
    """
    Calculate AUC using a many-hot vector.
    
    Parameters:
        predicted_scores (numpy array): Scores predicted by the model for all items.
        many_hot_vector (numpy array): Binary vector where 1 indicates a positive item and 0 indicates a negative item.
    
    Returns:
        float: The AUC value.
    """
    positive_scores = predicted_scores[many_hot_vector == 1]
    negative_scores = predicted_scores[many_hot_vector == 0]
    comparisons = positive_scores[:, None] > negative_scores[None, :]
    ties = positive_scores[:, None] == negative_scores[None, :]
    auc = (np.sum(comparisons) + 0.5 * np.sum(ties)) / (len(positive_scores) * len(negative_scores))
    return auc