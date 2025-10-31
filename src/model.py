import tensorflow as tf


class KerasModelBase(tf.keras.Model):
    """TF2/Keras 基类：所有推荐模型的基础类。"""

    def __init__(self, n_mid, embedding_dim, hidden_size, seq_len):
        super(KerasModelBase, self).__init__()
        self.n_mid = n_mid
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=n_mid,
            output_dim=embedding_dim,
            embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05),
            name="mid_embedding_var_keras"
        )
        self.item_bias = self.add_weight(
            name="bias_lookup_table_keras",
            shape=(n_mid,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=False,
        )

    def embed_items(self, item_ids):
        return self.item_embedding(item_ids)

    def get_item_embeddings(self):
        return self.item_embedding.weights[0]


class KerasModelDNN(KerasModelBase):
    def __init__(self, n_mid, embedding_dim, hidden_size, seq_len=256):
        super(KerasModelDNN, self).__init__(n_mid, embedding_dim, hidden_size, seq_len)
        self.proj = tf.keras.layers.Dense(hidden_size, activation=None)

    def call(self, inputs, training=False):
        # inputs: (mid, mid_hist, mask)
        mid, mid_hist, mask = inputs
        item_emb = self.embed_items(mid)
        hist_emb = self.embed_items(mid_hist)
        mask_f = tf.cast(mask, tf.float32)
        hist_emb_masked = hist_emb * tf.expand_dims(mask_f, -1)
        denom = tf.reduce_sum(mask_f, axis=1, keepdims=True) + 1e-9
        user_hist_mean = tf.reduce_sum(hist_emb_masked, axis=1) / denom
        user_vec = self.proj(user_hist_mean)
        return user_vec, item_emb

    def output_user(self, mid_hist, mask):
        user_vec, _ = self([None, mid_hist, mask], training=False)
        return user_vec

    def output_item(self):
        return self.get_item_embeddings()


class KerasModelGRU4REC(KerasModelBase):
    def __init__(self, n_mid, embedding_dim, hidden_size, seq_len=256):
        super(KerasModelGRU4REC, self).__init__(n_mid, embedding_dim, hidden_size, seq_len)
        self.gru = tf.keras.layers.GRU(hidden_size, return_sequences=False)

    def call(self, inputs, training=False):
        # inputs: (mid, mid_hist, mask)
        mid, mid_hist, mask = inputs
        item_emb = self.embed_items(mid)
        hist_emb = self.embed_items(mid_hist)
        mask_bool = tf.cast(mask, tf.bool)
        user_vec = self.gru(hist_emb, mask=mask_bool)
        return user_vec, item_emb

    def output_user(self, mid_hist, mask):
        user_vec, _ = self([None, mid_hist, mask], training=False)
        return user_vec

    def output_item(self):
        return self.get_item_embeddings()


class KerasCapsuleNetwork(tf.keras.layers.Layer):
    def __init__(self, dim, seq_len, bilinear_type=0, num_interest=4, hard_readout=True, relu_layer=True):
        super(KerasCapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        if self.relu_layer:
            self.proj_layer = tf.keras.layers.Dense(self.dim, activation=tf.nn.relu)
        # learnable bilinear weight for capsule
        if self.bilinear_type == 0:
            self.item_bilinear = tf.keras.layers.Dense(dim, use_bias=False)
        elif self.bilinear_type == 1:
            self.item_bilinear = tf.keras.layers.Dense(dim * num_interest, use_bias=False)
        elif self.bilinear_type == 2:
            self.w = self.add_weight(
                shape=(1, seq_len, num_interest * dim, dim),
                initializer=tf.keras.initializers.RandomNormal(),
                name="capsule_w",
                trainable=True,
            )

    def call(self, item_his_emb, item_eb, mask):
        # item_his_emb: [B, T, dim]  item_eb: [B, dim]  mask: [B, T]
        batch_size = tf.shape(item_his_emb)[0]
        if self.bilinear_type == 0:
            item_emb_hat = self.item_bilinear(item_his_emb)  # [B, T, dim]
            item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
        elif self.bilinear_type == 1:
            item_emb_hat = self.item_bilinear(item_his_emb)  # [B, T, dim*num_interest]
        else:
            # bilinear_type == 2
            w = self.w  # [1, T, H*D, D]
            u = tf.expand_dims(item_his_emb, 2)  # [B, T, 1, D]
            item_emb_hat = tf.reduce_sum(w[:, :self.seq_len, :, :] * u, axis=3)  # [B, T, H*D]

        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim])  # [B,T,H,D]
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])  # [B,H,T,D]
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.num_interest, self.seq_len, self.dim])

        item_emb_hat_iter = tf.stop_gradient(item_emb_hat)  # to match原TF版本 stop grad

        # routing
        capsule_weight = tf.zeros([batch_size, self.num_interest, self.seq_len], dtype=tf.float32)
        mask_expand = tf.expand_dims(mask, 1)  # [B,1,T]

        for i in range(3):
            atten_mask = tf.tile(mask_expand, [1, self.num_interest, 1])  # [B,H,T]
            paddings = tf.zeros_like(atten_mask)
            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.equal(atten_mask, 0), paddings, capsule_softmax_weight)  # mask
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)  # [B,H,1,T]
            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)  # [B,H,1,D]
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, keepdims=True)  # [B,H,1,1]
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule
                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)  # [B,H,1,D]
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, keepdims=True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.dim])  # [B,H,D]
        if self.relu_layer:
            interest_capsule = self.proj_layer(interest_capsule)
        # attention读出
        atten = tf.matmul(interest_capsule, tf.expand_dims(item_eb, -1))  # [B,H,1]
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self.num_interest]), 1))  # [B,H]
        if self.hard_readout:
            argmax_idx = tf.argmax(atten, axis=1, output_type=tf.int32)  # [B]
            batch_idx = tf.range(tf.shape(item_his_emb)[0]) * self.num_interest
            readout = tf.gather(tf.reshape(interest_capsule, [-1, self.dim]), argmax_idx + batch_idx)
        else:
            att_tmp = tf.reshape(atten, [batch_size, 1, self.num_interest])
            readout = tf.matmul(att_tmp, interest_capsule)  # [B,1,D]
            readout = tf.reshape(readout, [batch_size, self.dim])
        return interest_capsule, readout


class KerasModelMIND(KerasModelBase):
    def __init__(self, n_mid, embedding_dim, hidden_size, num_interest, seq_len=256, hard_readout=True,
                 relu_layer=True):
        super(KerasModelMIND, self).__init__(n_mid, embedding_dim, hidden_size, seq_len)
        self.num_interest = num_interest
        self.capsule = KerasCapsuleNetwork(hidden_size, seq_len, bilinear_type=0, num_interest=num_interest,
                                           hard_readout=hard_readout, relu_layer=relu_layer)

    def call(self, inputs, training=False):
        mid, mid_hist, mask = inputs
        item_emb = self.embed_items(mid)  # [B, D]
        hist_emb = self.embed_items(mid_hist)  # [B, T, D]
        interest_capsule, readout = self.capsule(hist_emb, item_emb, mask)
        # readout: [B,D]
        return readout, item_emb

    def output_user(self, mid_hist, mask):
        """返回所有兴趣向量用于评估（与TF1.x版本一致）"""
        # 使用一个dummy item来计算interest_capsule
        batch_size = tf.shape(mid_hist)[0]
        dummy_mid = tf.zeros((batch_size,), dtype=tf.int32)
        item_emb = self.embed_items(dummy_mid)
        hist_emb = self.embed_items(mid_hist)
        interest_capsule, _ = self.capsule(hist_emb, item_emb, mask)
        return interest_capsule  # [batch_size, num_interest, dim]

    def output_item(self):
        return self.get_item_embeddings()


class KerasModelComiRecDR(KerasModelBase):
    def __init__(self, n_mid, embedding_dim, hidden_size, num_interest, seq_len=256, hard_readout=True,
                 relu_layer=False):
        super(KerasModelComiRecDR, self).__init__(n_mid, embedding_dim, hidden_size, seq_len)
        self.num_interest = num_interest
        self.capsule = KerasCapsuleNetwork(hidden_size, seq_len, bilinear_type=2, num_interest=num_interest,
                                           hard_readout=hard_readout, relu_layer=relu_layer)

    def call(self, inputs, training=False):
        mid, mid_hist, mask = inputs
        item_emb = self.embed_items(mid)
        hist_emb = self.embed_items(mid_hist)
        interest_capsule, readout = self.capsule(hist_emb, item_emb, mask)
        # 训练时返回readout用于计算loss
        return readout, item_emb

    def output_user(self, mid_hist, mask):
        """返回所有兴趣向量用于评估（与TF1.x版本一致）"""
        # 使用一个dummy item来计算interest_capsule
        batch_size = tf.shape(mid_hist)[0]
        dummy_mid = tf.zeros((batch_size,), dtype=tf.int32)
        item_emb = self.embed_items(dummy_mid)
        hist_emb = self.embed_items(mid_hist)
        interest_capsule, _ = self.capsule(hist_emb, item_emb, mask)
        return interest_capsule  # [batch_size, num_interest, dim]

    def output_item(self):
        return self.get_item_embeddings()


class KerasModelComiRecSA(KerasModelBase):
    def __init__(self, n_mid, embedding_dim, hidden_size, num_interest, seq_len=256):
        super(KerasModelComiRecSA, self).__init__(n_mid, embedding_dim, hidden_size, seq_len)
        self.num_interest = num_interest
        self.ffn = tf.keras.layers.Dense(hidden_size * 4, activation=tf.nn.tanh)
        self.head_proj = tf.keras.layers.Dense(num_interest, activation=None)

    def call(self, inputs, training=False):
        mid, mid_hist, mask = inputs
        item_emb = self.embed_items(mid)
        hist_emb = self.embed_items(mid_hist)
        hidden = self.ffn(hist_emb)
        att_logits = self.head_proj(hidden)  # [B, T, H]
        att_logits = tf.transpose(att_logits, [0, 2, 1])  # [B, H, T]
        mask_f = tf.cast(mask, tf.float32)
        neg_inf = tf.ones_like(att_logits) * (-1e9)
        att_logits = tf.where(tf.equal(tf.expand_dims(mask_f, 1), 1.0), att_logits, neg_inf)
        att_w = tf.nn.softmax(att_logits, axis=-1)
        interest_emb = tf.matmul(att_w, hist_emb)  # [B, H, D]
        # 与目标交互选择兴趣
        sim = tf.matmul(interest_emb, tf.expand_dims(item_emb, -1))  # [B, H, 1]
        sim = tf.nn.softmax(tf.squeeze(sim, -1), axis=-1)  # [B, H]
        user_vec = tf.einsum('bh,bhd->bd', sim, interest_emb)  # [B, D]
        # 训练时返回单个user_vec用于计算loss
        return user_vec, item_emb

    def output_user(self, mid_hist, mask):
        """返回所有兴趣向量用于评估（与TF1.x版本一致）"""
        hist_emb = self.embed_items(mid_hist)
        hidden = self.ffn(hist_emb)
        att_logits = self.head_proj(hidden)  # [B, T, H]
        att_logits = tf.transpose(att_logits, [0, 2, 1])  # [B, H, T]
        mask_f = tf.cast(mask, tf.float32)
        neg_inf = tf.ones_like(att_logits) * (-1e9)
        att_logits = tf.where(tf.equal(tf.expand_dims(mask_f, 1), 1.0), att_logits, neg_inf)
        att_w = tf.nn.softmax(att_logits, axis=-1)
        interest_emb = tf.matmul(att_w, hist_emb)  # [B, H, D]
        return interest_emb  # [B, H, D]

    def output_item(self):
        return self.get_item_embeddings()
