# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from models.BaseModel import GeneralModel

class GFormer(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'batch_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=2, help='Number of LightGCN layers.')
        parser.add_argument('--num_layers_transformer', type=int, default=1, help='Number of Graph Transformer layers.')
        parser.add_argument('--n_heads', type=int, default=2, help='Number of attention heads.')
        parser.add_argument('--n_anchors', type=int, default=32, help='Number of anchor nodes.')
        parser.add_argument('--lambda1', type=float, default=1.0, help='Weight for Rationale Discovery Loss.')
        parser.add_argument('--lambda2', type=float, default=1e-3, help='Weight for Independence Loss.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.n_anchors = args.n_anchors
        
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        
        # 1. 构建图结构
        self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

        # 2. 定义参数 (先给个随机值，防止报错，真正初始化在 forward 里守门)
        self.user_emb = nn.Parameter(torch.randn(self.user_num, self.emb_size) * 0.01)
        self.item_emb = nn.Parameter(torch.randn(self.item_num, self.emb_size) * 0.01)
        
        # 3. Graph Transformer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.emb_size, 
            nhead=self.n_heads, 
            dim_feedforward=self.emb_size * 2,
            dropout=0.1,
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.num_layers_transformer)

        # 4. Anchor & Fusion
        self.anchor_embedding = nn.Parameter(torch.randn(self.n_anchors, self.emb_size) * 0.1)
        self.fusion_gate = nn.Linear(self.emb_size * 2, self.emb_size)

    @staticmethod
    def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1
        R = R.tolil()
        adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T
        adj_mat = adj_mat.todok()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1)) + 1e-10
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()
        if selfloop_flag:
            norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        else:
            norm_adj_mat = normalized_adj_single(adj_mat)
        return norm_adj_mat.tocsr()

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, feed_dict):
        # === 核武器：运行时强制初始化检测 ===
        # 只要发现全是0，当场重新初始化！
        if torch.abs(self.user_emb).sum() < 1e-5:
            # 这里的 print 应该只会在第一次 forward 时出现一次
            print("\n[WARN] 检测到 Embedding 被清零！正在执行 Forward 强制重置...")
            with torch.no_grad():
                nn.init.xavier_normal_(self.user_emb)
                nn.init.xavier_normal_(self.item_emb)
                nn.init.xavier_normal_(self.anchor_embedding)
            print(f"[INFO] 重置完成。User Mean Now: {self.user_emb.abs().mean().item():.6f}")

        # 1. LightGCN 传播
        ego_embeddings = torch.cat([self.user_emb, self.item_emb], 0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        local_emb = torch.stack(all_embeddings, dim=1).mean(dim=1)
        
        # 2. Graph Transformer
        trans_input = local_emb.unsqueeze(0) 
        global_trans_out = self.transformer_encoder(trans_input).squeeze(0)

        # 3. 融合
        combined = torch.cat([local_emb, global_trans_out], dim=1)
        final_emb = torch.sigmoid(self.fusion_gate(combined)) * local_emb + global_trans_out

        # 4. 取出
        user_ids = feed_dict['user_id']
        item_ids = feed_dict['item_id']

        u_g = final_emb[user_ids] 
        
        if item_ids.dim() == 2:
            i_g = final_emb[self.user_num + item_ids] 
            prediction = (u_g.unsqueeze(1) * i_g).sum(dim=-1)
        else:
            i_g = final_emb[self.user_num + item_ids]
            prediction = (u_g * i_g).sum(dim=-1)

        return {
            'prediction': prediction, 
            'user_emb': u_g, 
            'item_emb': i_g, 
            'local_emb': local_emb, 
            'global_emb': global_trans_out
        }

    def loss(self, out_dict: dict) -> torch.Tensor:
        predictions = out_dict['prediction']
        pos_scores = predictions[:, 0]
        neg_scores = predictions[:, 1]
        
        bpr_loss = -torch.log(1e-10 + torch.sigmoid(pos_scores - neg_scores)).mean()
        
        local_emb = out_dict['local_emb']
        global_emb = out_dict['global_emb']
        
        idx = torch.randint(0, local_emb.shape[0], (1024,)).to(self.device)
        l_sample = local_emb[idx]
        g_sample = global_emb[idx]
        
        score = F.cosine_similarity(l_sample, g_sample)
        cir_loss = torch.mean(torch.abs(score)) 

        rd_loss = torch.norm(global_emb, p=2) * 1e-4

        total_loss = bpr_loss + self.lambda1 * rd_loss + self.lambda2 * cir_loss
        
        return total_loss