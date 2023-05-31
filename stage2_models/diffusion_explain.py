
import torch
from .vq_diffusion import *


def alpha_schedule(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    """先根据线性方式定义 att(\bar \alpha_t) 和 ctt(\bar \gamma_t)
       然后根据公式计算出 btt(\beta_t, \bar \beta_t)
    """
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1 # \bar \alpha_t, 越来越小
    att = np.concatenate(([1], att))
    print("att: ", att)
    at = att[1:]/att[:-1]
    print("at: ", at)
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1 # \bar \gamma_t, 越来越大
    ctt = np.concatenate(([0], ctt))
    print("ctt: ", ctt)
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct  # \gamma_t
    bt = (1-at-ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1-att-ctt)/N
    print("att: ", att)
    print("btt: ", btt)
    print("ctt: ", ctt)
    return at, bt, ct, att, btt, ctt


N = 10
x_start = torch.LongTensor([[0,1,2,3,4], [2,3,9,9,9]])
log_x_start = index_to_log_onehot(x_start, N)
print("log_x_start: ", log_x_start.shape)

num_timesteps = 10



at, bt, ct, att, btt, ctt = alpha_schedule(num_timesteps, N=N)

at = torch.tensor(at.astype('float64'))
bt = torch.tensor(bt.astype('float64'))
ct = torch.tensor(ct.astype('float64'))
log_at = torch.log(at)
log_bt = torch.log(bt)
log_ct = torch.log(ct)
att = torch.tensor(att.astype('float64'))
btt = torch.tensor(btt.astype('float64'))
ctt = torch.tensor(ctt.astype('float64'))
log_cumprod_at = torch.log(att)
log_cumprod_bt = torch.log(btt)
log_cumprod_ct = torch.log(ctt)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # [bs] -> [bs, 1, 1]

t = torch.LongTensor([0,0])
t = (t + (num_timesteps + 1))%(num_timesteps + 1)


log_cumprod_at = extract(log_cumprod_at, t, log_x_start.shape)         # at~
log_cumprod_bt = extract(log_cumprod_bt, t, log_x_start.shape)         # bt~
log_cumprod_ct = extract(log_cumprod_ct, t, log_x_start.shape)         # ct~
log_1_min_cumprod_ct = extract(log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

log_probs = torch.cat(
            [   log_add_exp(log_x_start[:,:-2,:]+log_cumprod_at, log_cumprod_bt), # 前N-1个词
                log_add_exp(log_x_start[:,-2:-1,:]+log_1_min_cumprod_ct, log_cumprod_ct), # 第N个词，也就是mask,
                torch.log(torch.zeros_like(log_x_start[:, -1:, :]).fill_(1e-30))
            ],
            dim=1
        )

xt = log_onehot_to_index(log_probs)

print("log_probs: ", torch.exp(log_probs), torch.exp(log_probs).shape)
print("xt: ", xt)