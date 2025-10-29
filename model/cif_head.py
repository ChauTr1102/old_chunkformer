# model/cif_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CifCtcHead(nn.Module):
    def __init__(self, cif_dim: int, vocab_size: int, blank_id: int):
        super().__init__()
        self.fc = nn.Linear(cif_dim, vocab_size)
        self.blank_id = blank_id

    def forward_logits(self, cif_out):  # (B,Tc,Dc) -> (B,Tc,V)
        return self.fc(cif_out)

    def log_probs_TNC(self, cif_out):
        # return (T_c, B, V) for torch.ctc_loss
        logits = self.forward_logits(cif_out)          # (B,Tc,V)
        logp = F.log_softmax(logits, dim=-1)
        return logp.transpose(0, 1)                   # (Tc,B,V)

    @torch.no_grad()
    def greedy_ctc(self, cif_out):
        # returns list[int] per batch (after CTC collapse)
        logits = self.forward_logits(cif_out)          # (B,Tc,V)
        pred = logits.argmax(-1)                      # (B,Tc)
        hyps = []
        for row in pred:
            last = None
            out = []
            for t in row.tolist():
                if t != self.blank_id and t != last:
                    out.append(t)
                last = t
            hyps.append(out)
        return hyps
