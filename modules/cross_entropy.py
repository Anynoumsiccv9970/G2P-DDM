
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyCriterionWCustomMetrics(nn.Module):

    def __init__(self, padding_idx):
        super().__init__()
        self.padding_idx = padding_idx


    def forward(self, logits, target, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        lprobs = logits.log_softmax(dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)

        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss


class CandidatePenaltyCrossEntropyCriterion(nn.Module):
    """Applies a (1-p(x_nt)) loss to each negative target ('candidate') x_nt."""

    def __init__(self, padding_idx, rank_alpha, candidate_type):
        super().__init__()
        self.rank_alpha = rank_alpha
        self.candidate_type = candidate_type
        self.padding_idx= padding_idx

    def forward(self, logits, target, reduce=True):
        lprobs = logits.log_softmax(dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        
        nsentences = target.size(0)
        target = target.view(-1)

        # -- mle loss
        true_token_lprobs = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='none',
        )
        mle_loss = true_token_lprobs.sum()
        
        # -- custom loss
        # Maximize (1 - p(x_nt)) for negative target tokens x_nt (equivalently minimize -log(1-p(x_nt)))

        # - form negative targets
        with torch.no_grad():
            # E.g. DABCC | D | EFFGD => {A,B,C} are negative targets.
            if self.candidate_type == 'prev_context':
                # Make 'the triangle'.
                ctx_cands = target.unsqueeze(0).expand(target.size(0), target.size(0))
                ctx_cands_ = (ctx_cands.tril(-1) + 0)
                ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
                ctx_cands = ctx_cands.tril(-1) + ctx_cands_

                # Don't include the target for that timestep as a negative target.
                ctx_cands = ctx_cands.masked_fill(ctx_cands == target.unsqueeze(1), self.padding_idx)
                negative_targets = torch.zeros_like(lprobs).scatter_(1, ctx_cands, 1)
            else:
                raise NotImplementedError('candidate type %s' % self.candidate_type)
        # - compute loss
        one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-5)

        custom_loss = -torch.log(one_minus_probs) * negative_targets
        custom_loss = custom_loss.sum()

        loss = mle_loss + self.rank_alpha * custom_loss

        return loss, mle_loss, custom_loss


if __name__ == "__main__":
    padding_idx = 0
    target = torch.LongTensor([3,2,3,4,5, 6,7,8,9,0])

    ctx_cands = target.unsqueeze(0).expand(target.size(0), target.size(0))
    print("ctx_cands1: ", ctx_cands)
    ctx_cands_ = (ctx_cands.tril(-1) + padding_idx)
    print("ctx_cands2: ", ctx_cands_)
    ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
    print("ctx_cands3: ", ctx_cands_)
    ctx_cands = ctx_cands.tril(-1) + ctx_cands_
    print("ctx_cands4: ", ctx_cands)

    # Don't include the target for that timestep as a negative target.
    ctx_cands = ctx_cands.masked_fill(ctx_cands == target.unsqueeze(1), padding_idx)
    # negative_targets = torch.zeros_like(lprobs).scatter_(1, ctx_cands, 1)
    print("ctx_cands5: ",ctx_cands.shape, ctx_cands)