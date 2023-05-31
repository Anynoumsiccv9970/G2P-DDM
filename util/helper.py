
import torch
import numpy as np
from .dtw import dtw


# Find the best timing match between a reference and a hypothesis, using DTW
def calculate_dtw(references, hypotheses):
    """
    Calculate the DTW costs between a list of references and hypotheses

    :param references: list of reference sequences to compare against
    :param hypotheses: list of hypothesis sequences to fit onto the reference

    :return: dtw_scores: list of DTW costs
    """
    # Euclidean norm is the cost function, difference of coordinates
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    dtw_scores = []

    # Remove the BOS frame from the hypothesis
    # hypotheses = hypotheses[:, 1:]

    # For each reference in the references list
    for i, ref in enumerate(references):
        # Cut the reference down to the max count value
        _ , ref_max_idx = torch.max(ref[:, -1], 0)
        if ref_max_idx == 0: ref_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        ref_count = ref[:ref_max_idx,:-1].cpu().numpy()

        # Cut the hypothesis down to the max count value
        hyp = hypotheses[i]
        _, hyp_max_idx = torch.max(hyp[:, -1], 0)
        if hyp_max_idx == 0: hyp_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        hyp_count = hyp[:hyp_max_idx,:-1].cpu().numpy()

        # Calculate DTW of the reference and hypothesis, using euclidean norm
        d, cost_matrix, acc_cost_matrix, path = dtw(ref_count, hyp_count, dist=euclidean_norm)

        # Normalise the dtw cost by sequence length
        d = d/acc_cost_matrix.shape[0]

        dtw_scores.append(d)

    # Return dtw scores and the hypothesis with altered timing
    return dtw_scores


if __name__ == "__main__":
    a = torch.randn(5, 12)
    b = torch.randn(10, 12)
    calculate_dtw(a, b)
