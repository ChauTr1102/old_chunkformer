import torch
import torch.nn as nn


class CifMiddleware(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Load configurations
        self.cif_threshold = cfg.cif_threshold
        self.cif_output_dim = cfg.cif_embedding_dim
        self.encoder_embed_dim = cfg.encoder_embed_dim
        self.produce_weight_type = cfg.produce_weight_type
        self.conv_cif_width = cfg.conv_cif_width
        self.conv_cif_dropout = cfg.conv_cif_dropout
        self.apply_scaling = cfg.apply_scaling
        self.apply_tail_handling = cfg.apply_tail_handling
        self.tail_handling_firing_threshold = cfg.tail_handling_firing_threshold

        # Build weight generator
        if self.produce_weight_type == "dense":
            self.dense_proj = Linear(
                self.encoder_embed_dim, self.encoder_embed_dim).cuda()
            self.weight_proj = Linear(
                self.encoder_embed_dim, 1).cuda()
        elif self.produce_weight_type == "conv":
            self.conv = torch.nn.Conv1d(
                self.encoder_embed_dim,
                self.encoder_embed_dim,
                self.conv_cif_width,
                stride=1, padding=int(self.conv_cif_width / 2),
                dilation=1, groups=1,
                bias=True, padding_mode='zeros'
            ).cuda()
            self.conv_dropout = torch.nn.Dropout(
                p=self.conv_cif_dropout).cuda()
            self.weight_proj = Linear(
                self.encoder_embed_dim, 1).cuda()
        else:
            self.weight_proj = Linear(
                self.encoder_embed_dim, 1).cuda()

        # Build the final projection layer (if encoder_embed_dim is not equal to cif_output_dim)
        if self.cif_output_dim != self.encoder_embed_dim:
            self.cif_output_proj = Linear(
                self.encoder_embed_dim, self.cif_output_dim, bias=False).cuda()

    def forward(self, encoder_outputs, target_lengths=None, carry=None, flush_tail=True):
        """
        Args:
            encoder_outputs: dict
              - "encoder_raw_out": (B, T, C)
              - "encoder_padding_mask": (B, T)  # True tại vùng pad
            target_lengths: (B,)  # training (scaling)
            carry: None hoặc {"res_w": (B,), "res_h": (B, C)}  # residual weight/state mang sang lượt sau
            flush_tail: nếu True (và eval), sẽ ép xả tail nếu res_w > threshold (kết thúc luồng)
        Returns:
          {
            "cif_out": (B, Tc, C),
            "cif_out_padding_mask": (B, Tc)  # 1=valid, 0=pad
            "quantity_out": (B,),
            "carry": {"res_w": (B,), "res_h": (B, C)}
          }
        """

        # ===== Collect inputs =====
        encoder_raw_outputs = encoder_outputs["encoder_raw_out"]  # (B, T, C)
        encoder_padding_mask = encoder_outputs["encoder_padding_mask"]  # (B, T)
        device = encoder_raw_outputs.device
        B, T, C = encoder_raw_outputs.size()

        # ===== Produce weights =====
        if self.produce_weight_type == "dense":
            proj_out = self.dense_proj(encoder_raw_outputs)
            act_proj_out = torch.relu(proj_out)
            sig_input = self.weight_proj(act_proj_out)
            weight = torch.sigmoid(sig_input)
        elif self.produce_weight_type == "conv":
            conv_input = encoder_raw_outputs.permute(0, 2, 1)
            conv_out = self.conv(conv_input)
            proj_input = conv_out.permute(0, 2, 1)
            proj_input = self.conv_dropout(proj_input)
            sig_input = self.weight_proj(proj_input)
            weight = torch.sigmoid(sig_input)
        else:
            sig_input = self.weight_proj(encoder_raw_outputs)
            weight = torch.sigmoid(sig_input)
        # weight: (B, T, 1)

        not_padding_mask = ~encoder_padding_mask
        weight = weight.squeeze(-1) * not_padding_mask.int()  # (B, T)
        org_weight = weight

        # ===== Scaling (train) =====
        if self.training and self.apply_scaling and (target_lengths is not None):
            weight_sum = weight.sum(-1)  # (B,)
            normalize_scalar = (target_lengths / weight_sum).unsqueeze(-1)  # (B,1)
            weight = weight * normalize_scalar

        # ===== Prepare integrate & fire =====
        padding_start_id = not_padding_mask.sum(-1)  # (B,) số frame hợp lệ đầu tiên là [0..padding_start_id-1]

        # NEW: init residuals from carry
        if carry is not None:
            prev_w0 = carry.get("res_w", torch.zeros(B, device=device))
            prev_h0 = carry.get("res_h", torch.zeros(B, C, device=device))
        else:
            prev_w0 = torch.zeros(B, device=device)
            prev_h0 = torch.zeros(B, C, device=device)

        accumulated_weights = torch.zeros(B, 0, device=device)
        accumulated_states = torch.zeros(B, 0, C, device=device)
        fired_states = torch.zeros(B, 0, C, device=device)
        fire_mask_per_frame = torch.zeros(B, T, dtype=torch.bool, device=device)
        # ===== Integrate and fire =====
        for i in range(T):
            # dùng residual của lượt trước nếu i==0, ngược lại lấy bước trước đó
            prev_accumulated_weight = prev_w0 if i == 0 else accumulated_weights[:, i - 1]
            prev_accumulated_state = prev_h0 if i == 0 else accumulated_states[:, i - 1, :]

            # quyết định fire
            cur_is_fired = ((prev_accumulated_weight + weight[:, i]) >= self.cif_threshold).unsqueeze(-1)  # (B,1)
            fire_mask_per_frame[:, i] = cur_is_fired.squeeze(-1)
            # tính các đại lượng
            cur_weight = weight[:, i].unsqueeze(-1)  # (B,1)
            prev_accumulated_weight_u = prev_accumulated_weight.unsqueeze(-1)  # (B,1)
            remained_weight = torch.ones_like(prev_accumulated_weight_u,
                                              device=device) - prev_accumulated_weight_u  # (B,1)

            # accumulated weight/state tại bước i
            cur_accumulated_weight = torch.where(
                cur_is_fired,
                cur_weight - remained_weight,
                cur_weight + prev_accumulated_weight_u
            )  # (B,1)

            enc_i = encoder_raw_outputs[:, i, :]  # (B,C)
            cur_accumulated_state = torch.where(
                cur_is_fired.repeat(1, C),
                (cur_weight - remained_weight) * enc_i,
                prev_accumulated_state + cur_weight * enc_i
            )  # (B,C)

            # fired state tại bước i (nơi fire có vector, nơi khác là 0)
            cur_fired_state = torch.where(
                cur_is_fired.repeat(1, C),
                prev_accumulated_state + remained_weight * enc_i,
                torch.zeros(B, C, device=device)
            )  # (B,C)

            # Tail handling (giữ nguyên logic cũ của bạn)
            if (not self.training) and self.apply_tail_handling:
                cur_fired_state = torch.where(
                    (torch.full((B, C), i, device=device) ==
                     padding_start_id.unsqueeze(-1).repeat(1, C)),
                    torch.where(
                        cur_accumulated_weight.repeat(1, C) <= self.tail_handling_firing_threshold,
                        torch.zeros(B, C, device=device),
                        cur_accumulated_state / (cur_accumulated_weight + 1e-10)
                    ),
                    cur_fired_state
                )
            # if (not self.training) and self.apply_tail_handling:
            #     # last_valid = padding_start_id - 1 (clamp để tránh âm)
            #     last_valid = (padding_start_id - 1).clamp(min=0)
            #     # ma trận chỉ số i (B,C)
            #     i_mat = torch.full((B, C), i, device=device)
            #     # điều kiện: đang ở frame hợp lệ cuối cùng
            #     at_last_valid = (i_mat == last_valid.unsqueeze(-1).repeat(1, C))
            #
            #     cur_fired_state = torch.where(
            #         at_last_valid,
            #         torch.where(
            #             # nếu tích luỹ sau bước i (tức residual) <= ngưỡng -> bỏ
            #             cur_accumulated_weight.repeat(1, C) <= self.tail_handling_firing_threshold,
            #             torch.zeros(B, C, device=device),
            #             # nếu > ngưỡng -> chuẩn hoá và giữ (ép xả)
            #             cur_accumulated_state / (cur_accumulated_weight + 1e-10)
            #         ),
            #         cur_fired_state
            #     )
            # mask vùng pad về 0
            cur_fired_state = torch.where(
                (torch.full((B, C), i, device=device) >
                 padding_start_id.unsqueeze(-1).repeat(1, C)),
                torch.zeros(B, C, device=device),
                cur_fired_state
            )

            # push step
            accumulated_weights = torch.cat([accumulated_weights, cur_accumulated_weight], dim=1)  # (B,Tc)
            accumulated_states = torch.cat([accumulated_states, cur_accumulated_state.unsqueeze(1)], dim=1)  # (B,Tc,C)
            fired_states = torch.cat([fired_states, cur_fired_state.unsqueeze(1)], dim=1)  # (B,Tc,C)

        # ===== Extract outputs =====
        fired_marks = (torch.abs(fired_states).sum(-1) != 0.0).int()  # (B,Tc)
        fired_utt_length = fired_marks.sum(-1)  # (B,)
        fired_max_length = fired_utt_length.max().int() if B > 0 else torch.tensor(0, device=device)
        cif_outputs = torch.zeros(B, fired_max_length, C, device=device)

        def dynamic_partition(data: torch.Tensor, partitions: torch.Tensor, num_partitions=None):
            assert len(partitions.shape) == 1
            assert (data.shape[0] == partitions.shape[0])
            if num_partitions is None:
                num_partitions = int(torch.max(torch.unique(partitions)).item() + 1)
            return [data[partitions == index] for index in range(num_partitions)]

        for j in range(B):
            cur_utt_fired_mark = fired_marks[j, :]
            cur_utt_fired_state = fired_states[j, :, :]
            parts = dynamic_partition(cur_utt_fired_state, cur_utt_fired_mark, 2)
            cur_utt_output = parts[1] if len(parts) > 1 else torch.zeros(0, C, device=device)
            cur_len = cur_utt_output.size(0)
            if cur_len < fired_max_length:
                pad = torch.zeros(fired_max_length - cur_len, C, device=device)
                cur_utt_output = torch.cat([cur_utt_output, pad], dim=0)
            cif_outputs[j:j + 1, :, :] = cur_utt_output.unsqueeze(0)

        cif_out_padding_mask = (torch.abs(cif_outputs).sum(-1) != 0.0).int()  # (B,Tc)

        quantity_out = org_weight.sum(-1) if self.training else weight.sum(-1)

        # Optional proj
        if self.cif_output_dim != C:
            cif_outputs = self.cif_output_proj(cif_outputs)  # (B,Tc,C')

        # ===== NEW: compute and return residual carry =====
        # Lấy residual tại "bước hợp lệ cuối cùng" (index = padding_start_id-1, có clamp để an toàn)
        last_valid_idx = (padding_start_id - 1).clamp(min=0)  # (B,)
        res_w = accumulated_weights.gather(1, last_valid_idx.unsqueeze(1)).squeeze(1)  # (B,)
        idx_exp = last_valid_idx.view(B, 1, 1).expand(B, 1, C)
        res_h = accumulated_states.gather(1, idx_exp).squeeze(1)  # (B,C)

        carry_out = {"res_w": res_w, "res_h": res_h}

        # Ép xả tail nếu kết thúc luồng (eval) và bạn yêu cầu
        if (not self.training) and flush_tail:
            th = getattr(self, "tail_handling_firing_threshold", 0.4)
            fire_mask = (res_w > th)  # (B,)
            if fire_mask.any():
                # chuẩn hoá residual để phát thêm 1 embedding
                extra = res_h / res_w.clamp_min(1e-10).unsqueeze(1)  # (B,C)
                # append vào cif_outputs & mask theo batch (pad đến cùng Tc+1)
                # (đơn giản: nối theo chiều T cho từng batch rồi pad lại)
                # Ở đây ta ghép theo batch bằng cách tạo container mới:
                Tc = cif_outputs.size(1)
                cif_outputs = torch.cat([cif_outputs, torch.zeros(B, 1, cif_outputs.size(2), device=device)], dim=1)
                # set hàng nào fire -> ghi extra
                cif_outputs[fire_mask, Tc, :] = extra[fire_mask, :]
                # cập nhật mask
                extra_mask = torch.zeros(B, 1, dtype=torch.int, device=device)
                extra_mask[fire_mask, 0] = 1
                cif_out_padding_mask = torch.cat([cif_out_padding_mask, extra_mask], dim=1)

                # reset residual ở các batch đã flush
                res_w = torch.where(fire_mask, torch.zeros_like(res_w), res_w)
                res_h = torch.where(fire_mask.unsqueeze(1), torch.zeros_like(res_h), res_h)
                carry_out = {"res_w": res_w, "res_h": res_h}

        return {
            "cif_out": cif_outputs,
            "cif_out_padding_mask": cif_out_padding_mask,
            "quantity_out": quantity_out,
            "carry": carry_out,
            "fire_mask_per_frame": fire_mask_per_frame.int(),  # (B, T)
        }

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m