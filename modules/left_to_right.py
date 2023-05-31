import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import Tensor
import numpy as np


def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


class LeftToRight(object):
    def __init__(self, text_bos, text_pad, text_eos, text_embedding, decoder, token_num):
        super(LeftToRight, self).__init__()
        self.text_bos = text_bos
        self.text_pad = text_pad
        self.text_eos = text_eos
        self.text_embedding = text_embedding
        self.decoder = decoder
        self.token_num = token_num


    def generate(self, encoder_out, src_mask, max_output_length,
                 beam_size, alpha, n_best=1):
        """ In each decoding step, find the k most likely partial hypotheses.

        :param encoder_out: [bs, src_len, hidden_size]
        :param src_mask: [bs, 1, src_len]
        :param max_output_length:
        :param beam_size:
        :param alpha: `alpha` factor for length penalty
        :param n_best:
        :return:
        """
        assert beam_size > 0, "Beam size must be >0."
        assert n_best <= beam_size, "Can only return {} best hypotheses.".format(beam_size)

        bs = encoder_out.size(0)

        encoder_out = encoder_out.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(bs*beam_size, -1, encoder_out.size(-1))
        # print("src_mask: ", src_mask.shape)

        src_mask = src_mask.unsqueeze(1).repeat(1, beam_size, 1).view(bs*beam_size, src_mask.size(-1))
        # print("src_mask: ", src_mask.shape)

        # numbering elements in the batch
        batch_offset = torch.arange(
            bs, dtype=torch.long, device=encoder_out.device
        )  # size=[bs], [0,1,2,...,bs]
        # print("batch_offset: ", batch_offset)

        # numbering elements in the extended batch, i.e. beam size copies of each
        # batch element
        beam_offset = torch.arange(
            0, bs * beam_size, step=beam_size, dtype=torch.long, device=encoder_out.device
        )   # size=[bs], [0, beam, beam*2, ..., beam*(bs-1)]
        # print("beam_offset: ", beam_offset)


        # keeps track of the top beam size hypotheses to expand for each element
        # in the batch to be further decoded (that are still "alive")
        alive_seq = torch.full(
            [bs * beam_size, 1],
            self.text_bos,
            dtype=torch.long,
            device=encoder_out.device,
        )
        # print("alive_seq: ", alive_seq.shape)  # [bs * beam_size, 1]

        # Give full probability to the first beam on the first step.
        topk_log_probs = torch.zeros(bs, beam_size, device=encoder_out.device)  # [bs, beam]
        topk_log_probs[:, 1:] = float("-inf")  # [bs, beam]
        # print("topk_log_probs: ", topk_log_probs)

        hypotheses = [[] for _ in range(bs)]

        results = {
            "predictions": [[] for _ in range(bs)],
            "scores": [[] for _ in range(bs)],
            "gold_score": [0] * bs,
        }

        for step in range(max_output_length):
            # This decides which part of the predicted sentence we feed to the
            # decoder to make the next prediction.
            # For Transformer, we feed the complete predicted sentence so far.

            trg_input = alive_seq  # complete prediction so far
            trg_input_mask = trg_input.ne(self.text_pad)
            # expand current hypotheses
            # decode one single step
            # logits: logits for final softmax
            # pylint: disable=unused-variable
            trg_embed = self.text_embedding(trg_input, trg_input_mask)

            decoder_outputs = self.decoder(
                trg_embed=trg_embed,
                encoder_output=encoder_out,
                src_mask=src_mask,
                trg_mask=trg_input_mask)  # [bs*beam, cur_trg_len, vocab_size]

            # print("decoder_outputs: ", decoder_outputs.shape)
            logits = decoder_outputs[:, -1]  # [bs*beam, vocab_size]
            # if (step+1) % self.token_num != 0:
            #     logits[:, self.text_eos] = 0.
            # print("logits: ", logits.shape)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(1) # [bs*beam, vocab_size]

            # multiply probs by the beam probability (=add logprobs)
            # print("topk_log_probs: ", topk_log_probs.view(-1).unsqueeze(1))
            log_probs += topk_log_probs.view(-1).unsqueeze(1)  # [bs*beam, 1] + [bs*beam, vocab_size]
            # print("log_probs: ", log_probs.shape, log_probs[:, :5])


            curr_scores = log_probs.clone()
            # print("curr_scores: ", curr_scores.shape, curr_scores[:, :20])

            # compute length penalty
            if alpha > -1:
                length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
                curr_scores /= length_penalty
            else:
                length_penalty = None

            # flatten log_probs into a list of possibilities
            # TODO? reshape之后 index 的排布是以 vocab_size 为整体排布的
            curr_scores = curr_scores.reshape(-1, beam_size * decoder_outputs.size(-1))  # [bs, beam*vocab_size]
            # print("curr_scores: ", curr_scores.shape, curr_scores[:, :20])

            # pick currently best top k hypotheses (flattened order)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)  # [bs, beam], [bs, beam]
            # print("topk_ids: ", topk_ids.shape, topk_ids)

            if alpha > -1:
                # recover original log probs
                topk_log_probs = topk_scores * length_penalty
            else:
                topk_log_probs = topk_scores.clone()

            # reconstruct beam origin and true word ids from flattened order
            topk_beam_index = topk_ids.div(decoder_outputs.size(-1)).long()  # [bs, beam], TODO 这个是 beam 的 index，取整
            topk_ids = topk_ids.fmod(decoder_outputs.size(-1))  # [bs, beam], TODO 这个是取余数，对应在一个 vocab_size 中的 index
            # print("topk_beam_index: ", topk_beam_index)
            # print("topk_ids: ", topk_ids)


            # map beam_index to batch_index in the flat representation
            batch_index = topk_beam_index + beam_offset[:topk_beam_index.size(0)].unsqueeze(1)
            # print("batch_index: ", batch_index)

            select_indices = batch_index.view(-1)
            # print("select_indices: ", select_indices)
            # exit()

            # append latest prediction
            # TODO 这里需要选出 alive_seq
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1
            )  # batch_size*k x hyp_len [bs*beam, cur_trg_len]

            is_finished = topk_ids.eq(self.text_eos)  # [bs, beam]

            if step + 1 == max_output_length:
                is_finished.fill_(True)

            # end condition is whether the top beam is finished
            end_condition = is_finished[:, 0].eq(True) # [bs], TODO? top beam

            # save finished hypotheses
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))  # [bs, beam, cur_trg_len]
                for i in range(is_finished.size(0)): # bs
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(True)
                    finished_hyp = is_finished[i].nonzero().view(-1) # [beam]
                    for j in finished_hyp:
                        # Check if the prediction has more than one EOS.
                        # If it has more than one EOS, it means that the prediction should have already
                        # been added to the hypotheses, so you don't have to add them again.
                        if (predictions[i, j, 1:] == self.text_eos).nonzero().numel() < 2:
                            hypotheses[b].append(
                                (
                                    topk_scores[i, j],
                                    predictions[i, j, 1:],
                                )  # ignore start_token
                            )
                    # if the batch reached the end, save the n_best hypotheses
                    if end_condition[i]:
                        best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)

                non_finished = end_condition.eq(False).nonzero().view(-1)
                # if all sentences are translated, no need to go further
                # pylint: disable=len-as-condition
                if len(non_finished) == 0:
                    break
                # remove finished batches for the next step

                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(
                    -1, alive_seq.size(-1)
                )
            # reorder indices, outputs and masks
            select_indices = batch_index.view(-1)
            encoder_out = encoder_out.index_select(0, select_indices)
            src_mask = src_mask.index_select(0, select_indices)

        def pad_and_stack_hyps(hyps, pad_value):
            filled = (
                    np.ones((len(hyps), max([h.shape[0] for h in hyps])), dtype=int) * pad_value
            )
            for j, h in enumerate(hyps):
                for k, i in enumerate(h):
                    filled[j, k] = i
            return filled

        # from results to stacked outputs
        assert n_best == 1
        
        # only works for n_best=1 for now
        final_outputs = pad_and_stack_hyps(
            [r[0].cpu().numpy() for r in results["predictions"]], pad_value=self.text_pad
        )
        # print("final_outputs: ", final_outputs.shape, final_outputs)
        # exit()
        return final_outputs


if __name__ == "__main__":
    a = torch.LongTensor([[1,2,3,4,5], [2,3,4,5,6]])
    b = tile(a, 2)
    c = a.unsqueeze(1).repeat(1,2,1)
    print(b)
    print(c)
