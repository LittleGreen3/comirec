import os
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.nn.rnn_cell import GRUCell


class Model(object):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="DNN"):
        self.model_flag = flag
        self.reg = False
        self.batch_size = batch_size
        self.n_mid = n_mid
        self.neg_num = 10
        with tf.compat.v1.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.uid_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.mask = tf.compat.v1.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.target_ph = tf.compat.v1.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.compat.v1.placeholder(tf.float64, [])

        self.mask_length = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)

        # Embedding layer
        with tf.compat.v1.name_scope('Embedding_layer'):
            self.mid_embeddings_var = tf.compat.v1.get_variable("mid_embedding_var", [n_mid, embedding_dim],
                                                                trainable=True)
            self.mid_embeddings_bias = tf.compat.v1.get_variable("bias_lookup_table", [n_mid],
                                                                 initializer=tf.compat.v1.zeros_initializer(),
                                                                 trainable=False)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

        self.item_eb = self.mid_batch_embedded
        self.item_his_eb = self.mid_his_batch_embedded * tf.reshape(self.mask, (-1, seq_len, 1))

    def build_sampled_softmax_loss(self, item_emb, user_emb):
        self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias,
                                                              tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb,
                                                              self.neg_num * self.batch_size, self.n_mid))

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_his_batch_ph: inps[2],
            self.mask: inps[3],
            self.lr: inps[4]
        }
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        item_embs = sess.run(self.mid_embeddings_var)
        return item_embs

    def output_user(self, sess, inps):
        user_embs = sess.run(self.user_eb, feed_dict={
            self.mid_his_batch_ph: inps[0],
            self.mask: inps[1]
        })
        return user_embs

    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)


class Model_DNN(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_DNN, self).__init__(n_mid, embedding_dim, hidden_size,
                                        batch_size, seq_len, flag="DNN")

        masks = tf.concat([tf.expand_dims(self.mask, -1) for _ in range(embedding_dim)], axis=-1)

        self.item_his_eb_mean = tf.reduce_sum(self.item_his_eb, 1) / (
                tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)
        self.user_eb = tf.keras.layers.Dense(hidden_size, activation=None)(self.item_his_eb_mean)
        self.build_sampled_softmax_loss(self.item_eb, self.user_eb)


class Model_GRU4REC(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_GRU4REC, self).__init__(n_mid, embedding_dim, hidden_size,
                                            batch_size, seq_len, flag="GRU4REC")
        with tf.compat.v1.name_scope('rnn_1'):
            self.sequence_length = self.mask_length
            rnn_outputs, final_state1 = tf.compat.v1.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_his_eb,
                                                                    sequence_length=self.sequence_length,
                                                                    dtype=tf.float32,
                                                                    scope="gru1")

        self.user_eb = final_state1
        self.build_sampled_softmax_loss(self.item_eb, self.user_eb)


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape


class CapsuleNetwork(tf.keras.layers.Layer):
    def __init__(self, dim, seq_len, bilinear_type=2, num_interest=4, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True

    def call(self, item_his_emb, item_eb, mask):
        with tf.compat.v1.variable_scope('bilinear'):
            if self.bilinear_type == 0:
                item_emb_hat = tf.keras.layers.Dense(self.dim, activation=None, use_bias=False)(item_his_emb)
                item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
            elif self.bilinear_type == 1:
                item_emb_hat = tf.keras.layers.Dense(self.dim * self.num_interest, activation=None, use_bias=False)(
                    item_his_emb)
            else:
                w = tf.compat.v1.get_variable(
                    'weights', shape=[1, self.seq_len, self.num_interest * self.dim, self.dim],
                    initializer=tf.compat.v1.random_normal_initializer())
                # [N, T, 1, C]
                u = tf.expand_dims(item_his_emb, axis=2)
                # [N, T, num_caps * dim_caps]
                item_emb_hat = tf.reduce_sum(w[:, :self.seq_len, :, :] * u, axis=3)

        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim])
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.num_interest, self.seq_len, self.dim])

        if self.stop_grad:
            item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name='item_emb_hat_iter')
        else:
            item_emb_hat_iter = item_emb_hat

        if self.bilinear_type > 0:
            capsule_weight = tf.stop_gradient(tf.zeros([get_shape(item_his_emb)[0], self.num_interest, self.seq_len]))
        else:
            capsule_weight = tf.stop_gradient(
                tf.compat.v1.truncated_normal([get_shape(item_his_emb)[0], self.num_interest, self.seq_len],
                                              stddev=1.0))

        for i in range(3):
            atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
            paddings = tf.zeros_like(atten_mask)

            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.equal(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.dim])

        if self.relu_layer:
            interest_capsule = tf.keras.layers.Dense(self.dim, activation=tf.nn.relu, name='proj')(interest_capsule)

        atten = tf.matmul(interest_capsule, tf.reshape(item_eb, [-1, self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self.num_interest]), 1))

        if self.hard_readout:
            readout = tf.gather(tf.reshape(interest_capsule, [-1, self.dim]),
                                tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(
                                    tf.shape(item_his_emb)[0]) * self.num_interest)
        else:
            readout = tf.matmul(tf.reshape(atten, [get_shape(item_his_emb)[0], 1, self.num_interest]), interest_capsule)
            readout = tf.reshape(readout, [get_shape(item_his_emb)[0], self.dim])

        return interest_capsule, readout


class Model_MIND(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, hard_readout=True,
                 relu_layer=True):
        super(Model_MIND, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="MIND")

        item_his_emb = self.item_his_eb

        capsule_network = CapsuleNetwork(hidden_size, seq_len, bilinear_type=0, num_interest=num_interest,
                                         hard_readout=hard_readout, relu_layer=relu_layer)
        self.user_eb, self.readout = capsule_network(item_his_emb, self.item_eb, self.mask)

        self.build_sampled_softmax_loss(self.item_eb, self.readout)


class Model_ComiRec_DR(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, hard_readout=True,
                 relu_layer=False):
        super(Model_ComiRec_DR, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len,
                                               flag="ComiRec_DR")

        item_his_emb = self.item_his_eb

        capsule_network = CapsuleNetwork(hidden_size, seq_len, bilinear_type=2, num_interest=num_interest,
                                         hard_readout=hard_readout, relu_layer=relu_layer)
        self.user_eb, self.readout = capsule_network(item_his_emb, self.item_eb, self.mask)

        self.build_sampled_softmax_loss(self.item_eb, self.readout)

    # 注意：不重写output_user，使用基类方法返回user_eb（所有兴趣向量）
    # 评估时会使用多兴趣向量策略，这样可以覆盖用户的不同兴趣


class Model_ComiRec_SA(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, add_pos=True):
        super(Model_ComiRec_SA, self).__init__(n_mid, embedding_dim, hidden_size,
                                               batch_size, seq_len, flag="ComiRec_SA")

        self.dim = embedding_dim
        item_list_emb = tf.reshape(self.item_his_eb, [-1, seq_len, embedding_dim])

        if add_pos:
            self.position_embedding = \
                tf.get_variable(
                    shape=[1, seq_len, embedding_dim],
                    name='position_embedding')
            item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])
        else:
            item_list_add_pos = item_list_emb

        num_heads = num_interest
        with tf.compat.v1.variable_scope("self_atten", reuse=tf.compat.v1.AUTO_REUSE) as scope:
            item_hidden = tf.keras.layers.Dense(hidden_size * 4, activation=tf.nn.tanh)(item_list_add_pos)
            item_att_w = tf.keras.layers.Dense(num_heads, activation=None)(item_hidden)
            item_att_w = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_heads, 1])
            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
            item_att_w = tf.nn.softmax(item_att_w)

            interest_emb = tf.matmul(item_att_w, item_list_emb)

        self.user_eb = interest_emb

        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_list_emb)[0], self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_list_emb)[0], num_heads]), 1))

        self.readout = tf.gather(tf.reshape(self.user_eb, [-1, self.dim]),
                                 tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(
                                     tf.shape(item_list_emb)[0]) * num_heads)

        self.build_sampled_softmax_loss(self.item_eb, self.readout)

    # 注意：不重写output_user，使用基类方法返回user_eb（所有兴趣向量）
    # 评估时会使用多兴趣向量策略，这样可以覆盖用户的不同兴趣


# =========================
# Keras-style skeleton (TF2)
# =========================

class KerasModelBase(tf.keras.Model):
    """TF2/Keras 基类骨架：保持与现有模型相同的语义但采用 Keras 结构。
    暂不接入训练脚本；后续会将 train.py 迁移到使用该基类。
    """

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
        # item_embed/reference for loss/推理

    def call(self, inputs, training=False):
        mid, mid_hist, mask = inputs
        item_emb = self.embed_items(mid)  # [B, D]
        hist_emb = self.embed_items(mid_hist)  # [B, T, D]
        interest_capsule, readout = self.capsule(hist_emb, item_emb, mask)
        # readout: [B,D]
        return readout, item_emb

    def output_user(self, mid_hist, mask):
        user_vec, _ = self([None, mid_hist, mask], training=False)
        return user_vec

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

    def output_user_interests(self, mid_hist, mask):
        """返回所有兴趣向量用于评估（与TF1.x版本一致）"""
        # 使用一个dummy item来计算interest_capsule
        batch_size = tf.shape(mid_hist)[0]
        dummy_mid = tf.zeros((batch_size,), dtype=tf.int32)
        item_emb = self.embed_items(dummy_mid)
        hist_emb = self.embed_items(mid_hist)
        interest_capsule, _ = self.capsule(hist_emb, item_emb, mask)
        return interest_capsule  # [batch_size, num_interest, dim]

    def output_user(self, mid_hist, mask):
        # 评估时返回所有兴趣向量（与TF1.x版本一致）
        return self.output_user_interests(mid_hist, mask)

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

    def output_user_interests(self, mid_hist, mask):
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

    def output_user(self, mid_hist, mask):
        # 评估时返回所有兴趣向量（与TF1.x版本一致）
        return self.output_user_interests(mid_hist, mask)

    def output_item(self):
        return self.get_item_embeddings()
