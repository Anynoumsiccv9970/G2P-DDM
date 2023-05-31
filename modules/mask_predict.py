
from itertools import count
import torch
import torch.nn.functional as F

class MaskPredict(object):
    
    def __init__(self, decoding_iterations, token_num):
        super().__init__()
        self.iterations = decoding_iterations
        self.token_num = token_num

    def generate_fast(self, model, trg_tokens, word_feat, word_mask, points_mask, pad_idx, mask_idx):
        """points_emd
        """
        bsz, seq_len = trg_tokens.size()
        pad_mask = ~points_mask
        seq_lens = seq_len - pad_mask.sum(dim=1)
        # print("pad_mask: ", pad_mask[:, :15])
        # print("seq_lens: ", seq_lens)
        iterations = seq_len if self.iterations is None else self.iterations
        
        # print("before tgt_tokens: ", trg_tokens[:, :15])
        trg_tokens, token_probs, present_self = self.generate_non_autoregressive(model, trg_tokens, word_feat, word_mask)
        # print("after tgt_tokens: ", trg_tokens[:, :15])
        assign_single_value_byte(trg_tokens, pad_mask, pad_idx)
        assign_single_value_byte(token_probs, pad_mask, 1.0)
        #print("Initialization: ", convert_tokens(tgt_dict, trg_tokens[0]))
        
        for counter in range(1, iterations):
            # print("="*10 + "{}".format(counter) + "="*10)
            num_mask = (seq_lens.float() * (1.0 - (counter / iterations))).long()
            # print("num_mask: ", num_mask)

            assign_single_value_byte(token_probs, pad_mask, 1.0)
            mask_ind = self.select_worst(token_probs, num_mask)
            assign_single_value_long(trg_tokens, mask_ind, mask_idx)
            assign_single_value_byte(trg_tokens, pad_mask, pad_idx)

            #print("Step: ", counter+1)
            #print("Masking: ", convert_tokens(tgt_dict, trg_tokens[0]))
            # print("before tgt_tokens: ", trg_tokens[:, :15])
            decoder_out, present_self = model.decoder(
                trg_tokens=trg_tokens, 
                encoder_output=word_feat, 
                src_mask=word_mask, 
                trg_mask=points_mask, 
                mask_future=False, 
                window_mask_future=True, 
                window_size=self.token_num)
            new_trg_tokens, new_token_probs, all_token_probs = generate_step_with_prob(decoder_out)
            

            assign_multi_value_long(token_probs, mask_ind, new_token_probs)
            assign_single_value_byte(token_probs, pad_mask, 1.0)
            
            assign_multi_value_long(trg_tokens, mask_ind, new_trg_tokens)
            assign_single_value_byte(trg_tokens, pad_mask, pad_idx)
            # print("after tgt_tokens: ", trg_tokens[:, :15])
            #print("Prediction: ", convert_tokens(tgt_dict, trg_tokens[0]))
        lprobs = token_probs.log().sum(-1)
        return trg_tokens, present_self
    
    def generate_separate(self, model, tag_name, trg_tokens, word_feat, word_mask, trg_mask, pad_idx, mask_idx):
        """
        """
        # print(trg_tokens.shape, trg_mask.shape)
        
        bsz, seq_len, tok_num = trg_tokens.size()
        pad_mask = trg_mask.unsqueeze(-1).repeat(1, 1, tok_num)
        pad_mask = ~pad_mask

        seq_lens = seq_len * tok_num - pad_mask.sum(dim=-1).sum(dim=-1)
        # print("seq_lens: ", seq_lens)
        # print("pad_mask: ", pad_mask.shape, pad_mask)

        iterations = seq_len if self.iterations is None else self.iterations
        
        # print("before tgt_tokens: ", trg_tokens[:, :15])
        decoder_output = model.decoder[tag_name](
                trg_tokens=trg_tokens, 
                encoder_output=word_feat, 
                src_mask=word_mask, 
                trg_mask=trg_mask, 
                mask_future=False, 
                window_mask_future=False, 
                window_size=self.token_num, 
                tag_name=tag_name)
        # print("decoder_output: ", decoder_output.shape)
        
        probs = F.softmax(decoder_output, dim=-1)
        token_probs, trg_tokens = probs.max(dim=-1)

        # print("after tgt_tokens 1: ", trg_tokens.shape, trg_tokens[0, :15, :])

        assign_single_value_byte(trg_tokens, pad_mask, pad_idx)
        assign_single_value_byte(token_probs, pad_mask, 1.0)
        
        #print("Initialization: ", convert_tokens(tgt_dict, trg_tokens[0]))
        
        for counter in range(1, iterations):
            # print("="*10 + "{}".format(counter) + "="*10)
            num_mask = (seq_lens.float() * (1.0 - (counter / iterations))).long()
            # print("num_mask: ", num_mask)

            # assign low probality with mask_id
            assign_single_value_byte(token_probs, pad_mask, 1.0)

            bsz, tgt_len, tok_num = token_probs.size()
            # print("token_probs: ", token_probs.shape, token_probs[0, :5, :])
            # print("trg_tokens: ", trg_tokens.shape, trg_tokens[0, :5, :]) # bs, len ,5

            # trg_tokens = trg_tokens.view(bsz, -1)
            mask_ind = self.select_worst(token_probs.view(bsz, -1), num_mask)
            assign_single_value_long(trg_tokens.view(bsz, -1), mask_ind, mask_idx)

            # print("token_probs: ", token_probs.shape, token_probs[0, :5, :])
            # print("trg_tokens: ", trg_tokens.shape, trg_tokens[0, :5, :])
            # exit()
            assign_single_value_byte(trg_tokens, pad_mask, pad_idx)

            #print("Step: ", counter+1)
            #print("Masking: ", convert_tokens(tgt_dict, trg_tokens[0]))
            # print("before tgt_tokens: ", trg_tokens[:, :15])
            decoder_output = model.decoder[tag_name](
                trg_tokens=trg_tokens, 
                encoder_output=word_feat, 
                src_mask=word_mask, 
                trg_mask=trg_mask, 
                mask_future=False, 
                window_mask_future=False, 
                window_size=self.token_num, 
                tag_name=tag_name)
            probs = F.softmax(decoder_output, dim=-1)
            new_token_probs, new_trg_tokens = probs.max(dim=-1)
            
            assign_multi_value_long(token_probs.view(bsz, -1), mask_ind, new_token_probs.view(bsz, -1))
            assign_single_value_byte(token_probs, pad_mask, 1.0)
            
            assign_multi_value_long(trg_tokens.view(bsz, -1), mask_ind, new_trg_tokens.view(bsz, -1))
            assign_single_value_byte(trg_tokens, pad_mask, pad_idx)
        #     print("after tgt_tokens {}: ".format(counter), trg_tokens[:, :15, :])
        # exit()
        # lprobs = token_probs.log().sum(-1)
        return trg_tokens
    

    def select_worst(self, token_probs, num_mask):
        # print("token_probs: ", token_probs.shape, token_probs[0, :10]) # [bs, tgt_len, 5]
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        mask_ind = torch.stack(masks, dim=0)
        return mask_ind



def duplicate_encoder_out(encoder_out, bsz, beam_size):
    encoder_out['encoder_out'] = encoder_out['encoder_out'].unsqueeze(2).repeat(1, 1, beam_size, 1).view(-1, bsz * beam_size, encoder_out['encoder_out'].size(-1))
    if encoder_out['encoder_padding_mask'] is not None:
        encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].unsqueeze(1).repeat(1, beam_size, 1).view(bsz * beam_size, -1)


def generate_step_with_prob(out):
    probs = F.softmax(out, dim=-1)
    max_probs, idx = probs.max(dim=-1)
    return idx, max_probs, probs


def assign_single_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y


def assign_multi_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y.view(-1)[i.view(-1).nonzero()]


def assign_single_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y


def assign_multi_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y.view(-1)[i.view(-1)]


def convert_tokens(dictionary, tokens):
    return ' '.join([dictionary[token] for token in tokens])