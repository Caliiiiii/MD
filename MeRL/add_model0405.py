import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class Model(nn.Module):
    def __init__(self, args, plm):  
        super(Model, self).__init__()
        self.plm_config = plm.config
        self.num_classes = args.MODEL.num_classes
        self.plm = plm
        for param in self.plm.parameters():
            param.requires_grad = True
        self.dropout1 = nn.Dropout(args.MODEL.dropout)
        self.dropout2 = nn.Dropout(args.MODEL.dropout)
        self.first_last_avg = args.MODEL.first_last_avg
        self.use_context = args.DATA.use_context
        self.use_pos = args.DATA.use_pos
        self.cat_method = args.MODEL.cat_method
        self.new_cat_method = args.MODEL.new_cat_method
        self.use_gat = args.DATA.use_gat
        # self.use_intra = args.MODEL.use_intra
        if self.new_cat_method in ['abs', 'dot']:
            l_cnt = 1
        elif self.new_cat_method in ['cat', 'abs_dot']:
            l_cnt = 2
        elif self.new_cat_method in ['cat_abs', 'cat_dot']:
            l_cnt = 3
        else:
            l_cnt = 4
            

        self.MIP_linear = nn.Linear(in_features=args.MODEL.embed_dim * l_cnt, out_features=args.MODEL.embed_dim)
        self.SPV_linear = nn.Linear(in_features=args.MODEL.embed_dim * l_cnt, out_features=args.MODEL.embed_dim)
        if self.use_pos:
            self.fc = nn.Linear(in_features=args.MODEL.embed_dim * 3, out_features=args.MODEL.num_classes)
        else:
            # (768*2,2,True)
            self.fc = nn.Linear(in_features=args.MODEL.embed_dim * 2, out_features=args.MODEL.num_classes)

        self.dropout3 = nn.Dropout(args.MODEL.dropout)
        self._init_weights(self.fc)
        self._init_weights(self.MIP_linear)
        self._init_weights(self.SPV_linear)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.plm_config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()  # -- initializes all the biases to zero

    def forward(self, inputs):
        ids_ls, segs_ls, att_mask_ls, \
            ids_rs, att_mask_rs, segs_rs, \
            neighbor_mask, \
            = inputs  

        # get embeddings from the pretrained language model
        if self.use_context:
            out_l = self.plm(ids_ls, token_type_ids=segs_ls,
                             attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, token_type_ids=segs_rs,
                             attention_mask=att_mask_rs, output_hidden_states=True)  # output_hidden_states模型是否返回所有隐层状态.
        else:
            out_l = self.plm(ids_ls, attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, attention_mask=att_mask_rs, output_hidden_states=True)

        # (64, 135, 768)
        last_l = out_l.hidden_states[-1]  # last layer hidden states of the PLM
        first_l = out_l.hidden_states[1]  # first layer hidden states of the PLM

        # (64, 90, 768)
        last_r = out_r.hidden_states[-1]
        first_r = out_r.hidden_states[1]

        # -- use the average of the first and the last hidden layer of PLMs as word embeddings
        if self.first_last_avg:
            embed_l = torch.div(first_l + last_l, 2)
            embed_r = torch.div(first_r + last_r, 2)
        else:
            embed_l = last_l
            embed_r = last_r

        embed_l = self.dropout1(embed_l)  # [batch_size, max_left_len, embed_dim]
        embed_r = self.dropout2(embed_r)  # [batch_size, max_left_len, embed_dim] 

        # H_l ==> H_t for target;  H_r ==> H_b for basic meaning
        tar_mask_ls = (segs_ls == 1).long()  # - (64,135)
        tar_mask_rs = (segs_rs == 1).long()  # - (64,90)

        # unsqueeze增加维度，squeeze去掉维度
        H_t = torch.mul(tar_mask_ls.unsqueeze(2), embed_l)  
        # （16，70，1） * (16,70,768) == (16,70,768) 同上
        H_b = torch.mul(tar_mask_rs.unsqueeze(2), embed_r)

        h_c = torch.mean(embed_l, dim=1)  # context representation
        h_t = torch.mean(H_t, dim=1)
        h_b = torch.mean(H_b, dim=1)

        if self.use_gat:
            neighbor_mask = neighbor_mask / neighbor_mask.sum()
        
            feature = embed_l * neighbor_mask.unsqueeze(2)
          
            feature = feature.sum(dim=1).float()  
           

            h_t = h_t + feature

        # -----------------------------
        if self.new_cat_method == 'cat':
            h_mip = torch.cat((h_t, h_b), dim=-1)  # 768 + 200 + 768 = 1736
            h_spv = torch.cat((h_c, h_t), dim=-1)  # 768 + 200 + 768 = 1736
        elif self.new_cat_method == 'abs':
            h_mip = torch.abs(h_t - h_b)
            h_spv = torch.abs(h_c - h_t)
        elif self.new_cat_method == 'dot':
            h_mip = torch.mul(h_t, h_b)
            h_spv = torch.mul(h_c, h_t)
        elif self.new_cat_method == 'abs_dot':
            h_mip = torch.cat((torch.abs(h_t - h_b), torch.mul(h_t, h_b)), dim=-1)
            h_spv = torch.cat((torch.abs(h_c - h_t), torch.mul(h_c, h_t)), dim=-1)
        elif self.new_cat_method == 'cat_abs':
            h_mip = torch.cat((h_t, h_b, torch.abs(h_t - h_b)), dim=-1)
            h_spv = torch.cat((h_c, h_t, torch.abs(h_c - h_t)), dim=-1)
        elif self.new_cat_method == 'cat_dot':
            h_mip = torch.cat((h_t, h_b, torch.mul(h_t, h_b)), dim=-1)
            h_spv = torch.cat((h_c, h_t, torch.mul(h_c, h_t)), dim=-1)
        elif self.new_cat_method == 'cat_abs_dot': 
            h_mip = torch.cat((h_t, h_b, torch.abs(h_t - h_b), torch.mul(h_t, h_b)), dim=-1) # -- (16,3072)
            h_spv = torch.cat((h_c, h_t, torch.abs(h_c - h_t), torch.mul(h_c, h_t)), dim=-1) # -- (16,3072)


        h_mip = self.MIP_linear(h_mip)  # - 768
        h_spv = self.SPV_linear(h_spv)  # - 768
        if self.use_pos:
            pos_mask = (segs_rs == 3).long()
            H_p = torch.mul(pos_mask.unsqueeze(2), embed_r)
            h_p = torch.mean(H_p, dim=1)
            final = torch.cat((h_mip, h_spv, h_p), dim=-1)
        else:
            final = torch.cat((h_mip, h_spv), dim=-1)

        final = self.dropout3(final)  
        out = self.fc(final)  # [batch_size, num_classes] 
        return out


class MIP_Model(nn.Module):
    def __init__(self, args, plm):  
        super(MIP_Model, self).__init__()
        self.plm_config = plm.config
        self.num_classes = args.MODEL.num_classes
        self.plm = plm
        for param in self.plm.parameters():
            param.requires_grad = True
        self.dropout1 = nn.Dropout(args.MODEL.dropout)
        self.dropout2 = nn.Dropout(args.MODEL.dropout)
        self.first_last_avg = args.MODEL.first_last_avg
        self.use_context = args.DATA.use_context
        self.use_pos = args.DATA.use_pos
        self.cat_method = args.MODEL.cat_method
        self.new_cat_method = args.MODEL.new_cat_method
        self.use_gat = args.DATA.use_gat
        if self.new_cat_method in ['abs', 'dot']:
            l_cnt = 1
        elif self.new_cat_method in ['cat', 'abs_dot']:
            l_cnt = 2
        elif self.new_cat_method in ['cat_abs', 'cat_dot']:
            l_cnt = 3
        else:
            l_cnt = 4

        self.MIP_linear = nn.Linear(in_features=args.MODEL.embed_dim * l_cnt, out_features=args.MODEL.embed_dim)
        if self.use_pos:
            self.fc = nn.Linear(in_features=args.MODEL.embed_dim * 2, out_features=args.MODEL.num_classes)
        else:
            
            self.fc = nn.Linear(in_features=args.MODEL.embed_dim, out_features=args.MODEL.num_classes)

        self.dropout3 = nn.Dropout(args.MODEL.dropout)
        self._init_weights(self.fc)
        self._init_weights(self.MIP_linear)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.plm_config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()  # -- initializes all the biases to zero

    def forward(self, inputs):
        ids_ls, segs_ls, att_mask_ls, \
            ids_rs, att_mask_rs, segs_rs, \
            neighbor_mask, \
            = inputs  

        if self.use_context:
            out_l = self.plm(ids_ls, token_type_ids=segs_ls,
                             attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, token_type_ids=segs_rs,
                             attention_mask=att_mask_rs, output_hidden_states=True)  # output_hidden_states模型是否返回所有隐层状态.
        else:
            out_l = self.plm(ids_ls, attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, attention_mask=att_mask_rs, output_hidden_states=True)

        # (64, 135, 768)
        last_l = out_l.hidden_states[-1]  # last layer hidden states of the PLM
        first_l = out_l.hidden_states[1]  # first layer hidden states of the PLM

        # (64, 90, 768)
        last_r = out_r.hidden_states[-1]
        first_r = out_r.hidden_states[1]

        # -- use the average of the first and the last hidden layer of PLMs as word embeddings
        if self.first_last_avg:
            embed_l = torch.div(first_l + last_l, 2)  # - (64,135,768)
            embed_r = torch.div(first_r + last_r, 2)  # - (64,90,768)
        else:
            embed_l = last_l
            embed_r = last_r

        # - (64,135,768)
        embed_l = self.dropout1(embed_l)  # [batch_size, max_left_len, embed_dim]
        # - (64,90,768)
        embed_r = self.dropout2(embed_r)  # [batch_size, max_left_len, embed_dim] 

        # H_l ==> H_t for target;  H_r ==> H_b for basic meaning
        tar_mask_ls = (segs_ls == 1).long()  # - (64,135)
        tar_mask_rs = (segs_rs == 1).long()  # - (64,90)

        H_t = torch.mul(tar_mask_ls.unsqueeze(2), embed_l)  
        H_b = torch.mul(tar_mask_rs.unsqueeze(2), embed_r)

        h_t = torch.mean(H_t, dim=1)  # - (64,768)
        h_b = torch.mean(H_b, dim=1)  # - (64,768)

       
        if self.use_gat:
            neighbor_mask = neighbor_mask / neighbor_mask.sum()  # - (64,135)
         
            feature = embed_l * neighbor_mask.unsqueeze(2)  # - (64,135,768)
          
            feature = feature.sum(dim=1).float()  # - (64,768)
            h_t = h_t + feature  

        # -----------------------------
        if self.new_cat_method == 'cat':
            h_mip = torch.cat((h_t, h_b), dim=-1)  # 768 + 200 + 768 = 1736
        elif self.new_cat_method == 'abs':
            h_mip = torch.abs(h_t - h_b)
        elif self.new_cat_method == 'dot':
            h_mip = torch.mul(h_t, h_b)
        elif self.new_cat_method == 'abs_dot':
            h_mip = torch.cat((torch.abs(h_t - h_b), torch.mul(h_t, h_b)), dim=-1)
        elif self.new_cat_method == 'cat_abs':
            h_mip = torch.cat((h_t, h_b, torch.abs(h_t - h_b)), dim=-1)
        elif self.new_cat_method == 'cat_dot':
            h_mip = torch.cat((h_t, h_b, torch.mul(h_t, h_b)), dim=-1)
        elif self.new_cat_method == 'cat_abs_dot':  
            h_mip = torch.cat((h_t, h_b, torch.abs(h_t - h_b), torch.mul(h_t, h_b)), dim=-1)  # -- (16,3072)

        h_mip = self.MIP_linear(h_mip)  # - 768
        if self.use_pos:
            pos_mask = (segs_rs == 3).long()
            H_p = torch.mul(pos_mask.unsqueeze(2), embed_r)
            h_p = torch.mean(H_p, dim=1)
            final = torch.cat((h_mip, h_p), dim=-1)
        
        else:
            final = h_mip

        final = self.dropout3(final)  # -- (16,1536)
        out = self.fc(final)  
        return out

class SPV_Model(nn.Module):
    def __init__(self, args, plm): 
        super(SPV_Model, self).__init__()
        self.plm_config = plm.config
        self.num_classes = args.MODEL.num_classes
        self.plm = plm
        for param in self.plm.parameters():
            param.requires_grad = True
        self.dropout1 = nn.Dropout(args.MODEL.dropout)
        self.dropout2 = nn.Dropout(args.MODEL.dropout)
        self.first_last_avg = args.MODEL.first_last_avg
        self.use_context = args.DATA.use_context
        self.use_pos = args.DATA.use_pos
        self.cat_method = args.MODEL.cat_method
        self.new_cat_method = args.MODEL.new_cat_method
        self.use_gat = args.DATA.use_gat
        if self.new_cat_method in ['abs', 'dot']:
            l_cnt = 1
        elif self.new_cat_method in ['cat', 'abs_dot']:
            l_cnt = 2
        elif self.new_cat_method in ['cat_abs', 'cat_dot']:
            l_cnt = 3
        else:
            l_cnt = 4

        self.SPV_linear = nn.Linear(in_features=args.MODEL.embed_dim * l_cnt, out_features=args.MODEL.embed_dim)
        if self.use_pos:
            self.fc = nn.Linear(in_features=args.MODEL.embed_dim * 2, out_features=args.MODEL.num_classes)
        else:
            self.fc = nn.Linear(in_features=args.MODEL.embed_dim, out_features=args.MODEL.num_classes)

        self.dropout3 = nn.Dropout(args.MODEL.dropout)
        self._init_weights(self.fc)
        self._init_weights(self.SPV_linear)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.plm_config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()  # -- initializes all the biases to zero

    def forward(self, inputs):
        ids_ls, segs_ls, att_mask_ls, \
            ids_rs, att_mask_rs, segs_rs, \
            neighbor_mask, \
            = inputs 

        if self.use_context:
            out_l = self.plm(ids_ls, token_type_ids=segs_ls,
                             attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, token_type_ids=segs_rs,
                             attention_mask=att_mask_rs, output_hidden_states=True)  # output_hidden_states模型是否返回所有隐层状态.
        else:
            out_l = self.plm(ids_ls, attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, attention_mask=att_mask_rs, output_hidden_states=True)

        # (64, 135, 768)
        last_l = out_l.hidden_states[-1]  # last layer hidden states of the PLM
        first_l = out_l.hidden_states[1]  # first layer hidden states of the PLM

        # (64, 90, 768)
        last_r = out_r.hidden_states[-1]
        first_r = out_r.hidden_states[1]

        if self.first_last_avg:
            embed_l = torch.div(first_l + last_l, 2)  # - (64,135,768)
            embed_r = torch.div(first_r + last_r, 2)  # - (64,90,768)
        else:
            embed_l = last_l
            embed_r = last_r

        # - (64,135,768)
        embed_l = self.dropout1(embed_l)  # [batch_size, max_left_len, embed_dim]
        # - (64,90,768)
        embed_r = self.dropout2(embed_r)  # [batch_size, max_left_len, embed_dim] 

        # H_l ==> H_t for target;  H_r ==> H_b for basic meaning
        tar_mask_ls = (segs_ls == 1).long()
        H_t = torch.mul(tar_mask_ls.unsqueeze(2), embed_l)  

        # (64,768)
        h_c = torch.mean(embed_l, dim=1)  # context representation
        h_t = torch.mean(H_t, dim=1)
      
        if self.use_gat:
            neighbor_mask = neighbor_mask / neighbor_mask.sum()  # - (64,135)
          
            feature = embed_l * neighbor_mask.unsqueeze(2)  # - (64,135,768)
          
            feature = feature.sum(dim=1).float() 
         

            h_t = h_t + feature  

        # -----------------------------
        if self.new_cat_method == 'cat':
            h_spv = torch.cat((h_c, h_t), dim=-1)  # 768 + 200 + 768 = 1736
        elif self.new_cat_method == 'abs':
            h_spv = torch.abs(h_c - h_t)
        elif self.new_cat_method == 'dot':
            h_spv = torch.mul(h_c, h_t)
        elif self.new_cat_method == 'abs_dot':
            h_spv = torch.cat((torch.abs(h_c - h_t), torch.mul(h_c, h_t)), dim=-1)
        elif self.new_cat_method == 'cat_abs':
            h_spv = torch.cat((h_c, h_t, torch.abs(h_c - h_t)), dim=-1)
        elif self.new_cat_method == 'cat_dot':
            h_spv = torch.cat((h_c, h_t, torch.mul(h_c, h_t)), dim=-1)
        elif self.new_cat_method == 'cat_abs_dot':  
            h_spv = torch.cat((h_c, h_t, torch.abs(h_c - h_t), torch.mul(h_c, h_t)), dim=-1)  # -- (16,3072)

        h_spv = self.SPV_linear(h_spv)  # - 768
        if self.use_pos:
            pos_mask = (segs_rs == 3).long()
            H_p = torch.mul(pos_mask.unsqueeze(2), embed_r)
            h_p = torch.mean(H_p, dim=1)
            final = torch.cat((h_spv, h_p), dim=-1)
        else:
            final = h_spv

        final = self.dropout3(final)  
        out = self.fc(final)  
        return out

if __name__ == "__main__":
    pass
