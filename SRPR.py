import datetime
import os
import traceback

from backbone.backbone import Backbone
from datasets.dataloader import DataLoader
from datasets.dataloaderEmbedding import DataLoader as DataLoaderEmbedding
# from datasets.mini_imagenet_dataset_v2 import MiniImageNetDataLoader as MiniImageNetDataLoader_v2
from datasets.dataloader_image_embedding_clusters import MiniImageNetDataLoader as MiniImageNetDataLoader_v2
from tools.augmentations import *


def random_sample_support(s, s_label, avg=True):
    b, w, k, d = tf.unstack(tf.shape(s))
    begin_end = tf.random.shuffle(tf.range(k + 1))[:2]
    begin, end = tf.reduce_min(begin_end), tf.reduce_max(begin_end)
    sub_s = s[..., begin:end, :]
    sub_label = s_label[..., begin:end, :]
    if avg:
        mean_s = tf.reduce_mean(sub_s, 2, keepdims=True)
        mean_s_label = tf.reduce_mean(sub_label, 2, keepdims=True)
        return mean_s, mean_s_label
    else:
        return sub_s, sub_label


def random_sample_support2(s):
    b, w, k, d = tf.unstack(tf.shape(s))
    W = tf.random.uniform([b, w, k, 1], 1., 10.)
    W = W / tf.reduce_sum(W, 2, keepdims=True)
    W = tf.cast(W, s.dtype)
    mean_s = tf.reduce_sum(s * W, 2)
    return mean_s


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                            self.learning_rate_base - self.warmup_learning_rate
                    ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


model_name = os.path.basename(__file__).split(".")[0]


class PrintCallback(tf.keras.callbacks.Callback):
    """Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def __init__(self, **kwargs):
        super(PrintCallback, self).__init__()
        self.kwargs = kwargs

    def on_epoch_end(self, epoch, logs=None):
        tf.print(self.model.predictor.T.numpy(), self.model.predictor.T2.numpy(), self.model.predictor.T3.numpy(),
                 self.model.predictor.numIter)


class QuickGELU(tf.keras.layers.Layer):
    def __init__(self, name="QuickGELU"):
        super(QuickGELU, self).__init__(name=name)

    def call(self, x: tf.Tensor):
        return x * tf.sigmoid(1.702 * x)


class Predictor(tf.keras.layers.Layer):
    def __init__(self, num_matrix, num_base, num_expressive, numIter=10, name=None, **kwargs):
        super(Predictor, self).__init__(name=name)
        self.num_expressive = num_expressive
        self.num_matrix = num_matrix
        self.num_base = num_base
        self.numIter = numIter
        self.hiddenDim = 256

    def build(self, input_shape):
        self.T = self.add_weight(
            'T',
            shape=[1],
            initializer=tf.constant_initializer([20.]),
            trainable=False)

        self.QueryEmbedding = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.hiddenDim),
            QuickGELU(),
            tf.keras.layers.Activation("softmax")
        ], name="QueryEmbedding")
        self.QueryEmbedding.build([None, self.num_matrix])

        self.ln = tf.keras.layers.LayerNormalization(name="ln")
        self.ln.build([None, input_shape[-1]])

        self.B = self.add_weight(
            shape=(self.hiddenDim, self.num_expressive, input_shape[-1]),
            initializer="he_normal",
            trainable=True,
            name="B"
        )

        self.positional_embedding = self.add_weight(
            shape=(self.numIter, input_shape[-1]),
            initializer="he_normal",
            trainable=True,
            name="positional_embedding"
        )

        self.embeddingAnchors = self.add_weight(
            'anchors',
            shape=[self.num_matrix, input_shape[-1]],
            initializer="he_normal",
            trainable=True)

        self.T2 = self.add_weight(
            'T2',
            shape=[1],
            initializer=tf.constant_initializer([1.]),
            # initializer="he_normal",
            trainable=False)
        self.T3 = self.add_weight(
            'T3',
            shape=[1],
            initializer=tf.constant_initializer([1.]),
            # initializer="he_normal",
            trainable=True)

        super(Predictor, self).build(input_shape)

    def clc(self, input, anchor, T=None):
        logits = tf.matmul(tf.nn.l2_normalize(input, -1)
                           , tf.nn.l2_normalize(anchor, -1), transpose_b=True)
        if T is None:
            T = self.T
        pred = tf.nn.softmax(logits * T, -1)
        return pred

    def get_config(self):
        config = {"num_matrix": self.num_matrix,
                  "num_expresive": self.num_expresive,
                  "num_base": self.num_base,
                  "numIter": self.numIter,
                  }
        base_config = super(Predictor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def queryParams(self, y, training=None):
        out = tf.einsum("bc, cd->bd", y, self.B)
        return out

    def querySelect(self, reference, T=1, training=None):
        x = tf.reshape(reference, [-1, tf.shape(reference)[-1]])
        # x = tf.matmul(x, self.embeddingAnchors, transpose_b=True).
        x = tf.matmul(x, tf.nn.l2_normalize(self.embeddingAnchors, -1), transpose_b=True)
        x = tf.nn.softmax(x * T, -1)
        x = self.QueryEmbedding(x, training=training)
        return x

    def update(self, P, G, scale, beta=0.05, training=None):
        sim = 1. - tf.reduce_sum(P * G, -1, keepdims=True)
        sim = tf.maximum(0., sim - beta)
        # sim = sim * tf.cast(tf.greater(sim, 0.05), tf.float32)
        lr = scale * sim
        P = tf.nn.l2_normalize(P + lr * G, -1)
        return P

    def updateAnchors(self, anchors):
        print("updateAnchors shape:", anchors.shape)
        self.embeddingAnchors.assign(anchors)

    def getIntraInter(self, S, S_label, norm=False):
        batch, ways, shots, dims = tf.unstack(tf.shape(S))
        if norm:
            S = tf.nn.l2_normalize(S, -1)
        Sum = tf.multiply(tf.expand_dims(S, -2),
                          tf.expand_dims(S_label, -1))

        Sum = tf.reduce_sum(Sum, [1, 2])

        Intra = tf.reshape(Sum, [batch, ways, 1, -1]) \
                - tf.cast(tf.greater(shots, 1), tf.float32) * S
        Intra = tf.reshape(Intra,
                           [batch, ways, shots, 1, -1]) \
                * tf.expand_dims(S_label, -1)
        Intra = tf.nn.l2_normalize(tf.reduce_sum(Intra, -2), -1)

        Inter = tf.reshape(Sum, [batch, 1, 1, ways, -1]) \
                * tf.expand_dims(1. - S_label, -1)
        Inter = tf.nn.l2_normalize(tf.reduce_sum(Inter, -2), -1)
        Pairs = tf.stack([Intra, Inter], -2)

        return Intra, Inter, Pairs

    def call(self, S, S_label=None, training=None, iterNum=10):
        batch, ways, shots, dims = tf.unstack(tf.shape(S))
        S = tf.nn.l2_normalize(S, -1)
        if training is True:
            P = random_sample_support2(S)
        else:
            P = tf.reduce_mean(S, 2)
        P = tf.reshape(P, [batch, ways, dims])

        def condition(step, S_input, P_input, P_list, S_list, V_list):
            return step < iterNum

        def body(step, S_input, P_input, P_list, S_list, V_list):
            # S_input = self.update(S_input, tf.expand_dims(P_input,-2), 0.5, training=training)
            y = self.querySelect(S_input, self.T2, training=training)
            V = tf.einsum("bc, cnd->bnd", y, self.B)
            V = self.ln(tf.reshape(V, [-1, tf.shape(V)[-1]]), training=training)
            V = tf.reshape(V, [*tf.unstack(tf.shape(S_input)[:2]), -1, self.num_expressive, tf.shape(V)[-1]])
            V = tf.nn.l2_normalize(V, -1)
            V = tf.reduce_mean(V, [-2], keepdims=False)
            V = tf.nn.l2_normalize(V, -1)

            S_input = self.update(S_input, V, 0.5, training=training)
            # S_input = self.update(S_input, V, 1, training=training)
            P_ = tf.reduce_mean(S_input, -2)
            P_ = tf.nn.l2_normalize(P_, -1)
            P_input = self.update(P_input, P_, 1, training=training)

            P_list = P_list.write(step, P_input)
            S_list = S_list.write(step, S_input)
            V_list = V_list.write(step, V)
            return step + 1, S_input, P_input, P_list, S_list, V_list

        if iterNum > 0:
            P_list = tf.TensorArray(dtype=tf.float32, size=iterNum)
            S_list = tf.TensorArray(dtype=tf.float32, size=iterNum)
            V_list = tf.TensorArray(dtype=tf.float32, size=iterNum)

            _, S_, P_out, P_list, S_list, V_list = tf.while_loop(condition, body,
                                                                 [0, S, P, P_list, S_list, V_list],
                                                                 None)
            P_list = P_list.stack()
            S_list = S_list.stack()
            V_list = V_list.stack()

            P_list.set_shape([P_list.shape[0], S.shape[0], S.shape[1], S.shape[-1]])
            S_list.set_shape([S_list.shape[0], S.shape[0], S.shape[1], S.shape[2], S.shape[-1]])
            V_list.set_shape([V_list.shape[0], S.shape[0], S.shape[1], S.shape[2], S.shape[-1]])
            # V_list.set_shape([V_list.shape[0], S.shape[0], S.shape[1], 1, S.shape[-1]])
            S_.set_shape([S.shape[0], S.shape[1], S.shape[2], S.shape[-1]])
            P_out.set_shape([P.shape[0], P.shape[1], P.shape[-1]])
            P_list = tf.concat([tf.reshape(P, [1, batch, ways, dims]), P_list], 0)
            S_list = tf.concat([tf.reshape(S, [1, batch, ways, -1, dims]), S_list], 0)


        else:
            P = tf.reduce_mean(S, 2)
            P = tf.nn.l2_normalize(P, -1)
            P_out = tf.reshape(P, [batch, ways, dims])
            P_list = tf.reshape(P, [1, batch, ways, dims])
            S_list = tf.reshape(S, [1, batch, ways, -1, dims])

        if training is True:
            return S_list, P_list, V_list
        else:
            return S_list, P_list

    def infer(self, S, Q, iterNum=10, training=None):
        S, S_label = S
        batch = tf.shape(S)[0]
        ways = tf.shape(S)[1]
        dims = tf.shape(S)[-1]
        SShape = tf.shape(S)
        QShape = tf.shape(Q)

        _, P = self(tf.reshape(S, SShape), S_label=S_label, iterNum=iterNum, training=training)
        test_pred = self.clc(tf.reshape(Q, [1, batch, -1, tf.shape(Q)[-1]]),
                             tf.reshape(P, [iterNum + 1, batch, ways, -1]))
        return test_pred


class FSLModel(tf.keras.Model):
    def __init__(self, imageshape=(84, 84, 3), num_class=64, num_base_embbeding=5, num_expresive=20, anchor_num=1,
                 name=model_name, backbone="resnet_12"):
        super(FSLModel, self).__init__(name=name)

        self.num_class = num_class
        self.encoder = Backbone(backbone, input_shape=imageshape, pooling=None, use_bias=False).get_model()
        self.encoder.build([None, *imageshape])
        index = 0
        feature_size_h, feature_size_w, feature_dim = [*self.encoder.output.shape[1:]]
        self.feature_dim = feature_dim

        self.last_max_pooling = tf.keras.layers.MaxPool2D(padding="same", name="last_max_pooling")
        self.last_max_pooling.build([None, feature_size_h, feature_size_w, feature_dim])
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="gap")
        self.gap.build([None, feature_size_h, feature_size_w, feature_dim])
        self.gmp = tf.keras.layers.GlobalMaxPooling2D(name="gmp")
        self.gmp.build([None, feature_size_h, feature_size_w, feature_dim])

        # def mix(x):
        #     x1 = self.gap(x)
        #     x2 = self.gap(self.last_max_pooling(x))
        #     return x1 + x2
        def mix(x):
            x = self.last_max_pooling(x)
            # x = tf.nn.l2_normalize(x)
            x = self.gap(x)
            return x
            return tf.pad(x, [[0, 0], [0, 768 - feature_dim]])

        self.mixGap = tf.keras.layers.Lambda(mix)
        self.mixGap.build([None, feature_size_h, feature_size_w, feature_dim])

        self.iterNum = 20
        self.iterNum_1s = self.iterNum

        self.predictor = Predictor(self.num_class * anchor_num, num_base_embbeding, num_expresive, self.iterNum,
                                   name="Predictor")
        self.predictor.build([None, None, None, feature_dim])

        self.build([None, *imageshape])
        self.summary()
        self.acc = tf.keras.metrics.CategoricalAccuracy(name="acc")
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss = tf.keras.metrics.Mean(name="r_loss")
        self.intra_metric = tf.keras.metrics.Mean(name="intra_loss")
        self.inter_metric = tf.keras.metrics.Mean(name="inter_loss")
        self.entropy_metric = tf.keras.metrics.Mean(name="entropy")
        self.InfoNCE_metric = tf.keras.metrics.Mean(name="InfoNCE")

        self.query_loss_metric = tf.keras.metrics.Mean("query_loss")
        self.mean_query_acc = tf.keras.metrics.Mean(name="mq_acc")
        self.mean_query_acc_1s = tf.keras.metrics.Mean(name="mq_acc1s")

        self.mean_query_acc_group = [tf.keras.metrics.Mean(name="mq_acc_{}".format(index)) for index in
                                     range(self.iterNum + 1)]
        self.mean_query_acc_1s_group = [tf.keras.metrics.Mean(name="mq_acc_1s_{}".format(index)) for index in
                                        range(self.iterNum_1s + 1)]
        self.mean_query_acc_base = tf.keras.metrics.Mean(name="mq_acc_base")
        self.mean_query_acc_one_shot = tf.keras.metrics.Mean(name="mq_acc_base_1s")
        self.T = 1. / 0.05
        self.alfa = 1.
        # self.T = 1. / 1.
        # self.T = 1. / 0.2
        # self.T = 1. / 0.1
        # self.T = 1. / 0.02
        # self.T = 1. / 0.01
        print("T:", self.T)

        scheduled_lrs = WarmUpCosine(
            learning_rate_base=0.0001,
            total_steps=10,
            warmup_learning_rate=0.0,
            warmup_steps=3,
        )
        self.opt = tf.keras.optimizers.SGD(0.001, momentum=0.9, nesterov=True, name="opt")

    def call(self, inputs, training=None, pool=True):
        features = self.encoder(inputs, training=training)
        if pool:
            return self.mixGap(features)
        return features

    def reset_metrics(self):
        # Resets the state of all the metrics in the model.
        for m in self.metrics:
            m.reset_states()

        self.acc.reset_states()
        self.loss_metric.reset_states()
        self.reconstruction_loss.reset_states()
        self.intra_metric.reset_states()
        self.inter_metric.reset_states()
        self.entropy_metric.reset_states()
        self.InfoNCE_metric.reset_states()

        self.query_loss_metric.reset_states()
        self.mean_query_acc.reset_states()
        self.mean_query_acc_1s.reset_states()

        self.mean_query_acc_base.reset_states()
        self.mean_query_acc_one_shot.reset_states()

        for m in self.mean_query_acc_group:
            m.reset_states()
        for m in self.mean_query_acc_1s_group:
            m.reset_states()

    def random_enrode(self, x):
        x = tf.reshape(x, tf.concat([[-1], tf.shape(x)[-3:]], 0))
        batch, h, w, c = tf.unstack(tf.shape(x))
        for _ in range(1):
            features = self.encoder(x, training=False)

            self_attention = self.self_attention_referenced_conv(features, training=False)
            self_attention = tf.image.resize(self_attention, [h, w])
            bg = x * (1. - self_attention)
            bg_mean = tf.reduce_mean(bg, [1, 2], keepdims=True)
            ratio = 0.
            bg_mean = tf.broadcast_to(bg_mean, tf.shape(x)) * self_attention
            keep_ratio = 0.005
            x = x * keep_ratio + (1. - keep_ratio) * tf.clip_by_value(
                x * (1. - self_attention) + x * ratio + bg_mean * (1 - ratio), 0., 1.)

        return x

    def S_step(self, data):
        support, query = data
        S, S_label, support_label_global = support
        Q, Q_label, _ = query
        # S, GS = S[:, 0, ...], S[:, 1, ...]
        # Q, GQ = Q[:, 0, ...], Q[:, 1, ...]
        batch = tf.shape(S)[0]
        ways = tf.shape(S)[1]
        shots = tf.shape(S)[2]
        dims = tf.shape(S)[3]
        query_shots = tf.shape(Q)[2]

        S = tf.reshape(S, tf.concat([[batch, ways, shots], tf.shape(S)[-1:]], 0))
        Q = tf.reshape(Q, tf.concat([[batch, ways, query_shots], tf.shape(Q)[-1:]], 0))
        S = tf.nn.l2_normalize(S, -1)
        Q = tf.nn.l2_normalize(Q, -1)
        Z = tf.concat([S, Q], 2)
        P_label = tf.reduce_mean(S_label, 2)
        Z_label = tf.concat([S_label, Q_label], 2)

        training = True
        iterNum = self.iterNum
        SShape = tf.shape(S)
        ZShape = tf.shape(Z)

        ProtoList = []

        step = tf.maximum(tf.shape(Z)[-2] // iterNum, 20)
        for it in range(1, iterNum + 1):
            ProtoList.append(tf.reduce_mean(Z[:, :, : step * it, :], 2))
        ProtoList = tf.stack(ProtoList, 0)
        target = tf.nn.l2_normalize(tf.reduce_mean(tf.nn.l2_normalize(Z, -1), 2), -1)

        with tf.GradientTape() as tape:
            S_list, P_list, V_list = self.predictor(S=S,
                                                    S_label=S_label,
                                                    training=training, iterNum=iterNum)
            Q_Pred = self.predictor.clc(tf.reshape(Q, [batch, -1, dims]),
                                        tf.reshape(P_list[-1], [batch, ways, dims]))
            meta_contrast_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    tf.reshape(Q_label, [batch, -1, tf.shape(Q_label)[-1]]),
                    tf.reshape(Q_Pred, [batch, -1, tf.shape(Q_Pred)[-1]])),
                -1)

            # Z_pred = self.predictor.clc(
            #     tf.repeat(tf.reshape(tf.stop_gradient(Z), [1, batch, -1, tf.shape(Z)[-1]]), iterNum, 0),
            #     tf.reshape(P_list[1:], [iterNum, batch, ways, dims]))
            #
            # meta_contrast_loss = tf.keras.losses.categorical_crossentropy(
            #     tf.repeat(tf.reshape(Z_label, [1, batch, -1, tf.shape(Z_label)[-1]]), iterNum, 0),
            #     tf.reshape(Z_pred, [iterNum, batch, -1, tf.shape(Z_pred)[-1]]))
            # meta_contrast_loss = tf.reduce_mean(meta_contrast_loss, 0)

            sim = 1. + tf.losses.cosine_similarity(tf.reshape(ProtoList, [iterNum, batch, ways, 1, -1]),
                                                   S_list[1:], -1)
            simTarget = 1. + tf.losses.cosine_similarity(tf.reshape(target, [1, batch, ways, 1, -1]),
                                                         S_list[1:], -1)
            simCmp = 1. + tf.losses.cosine_similarity(tf.reshape(target, [1, batch, ways, 1, -1]),
                                                      tf.reshape(ProtoList, [iterNum, batch, ways, 1, -1]), -1)
            mask = tf.cast(tf.less(simCmp, simTarget), tf.float32)
            sim = mask * sim
            recon_loss = tf.reduce_mean(sim, 0)
            meta_loss = tf.reduce_mean(meta_contrast_loss)
            lossRec = tf.reduce_mean(recon_loss)

            # l2B = tf.reshape(tf.nn.l2_normalize(self.predictor.B, -1),
            #                  [-1, tf.shape(self.predictor.B)[-1]])
            # sim = tf.matmul(l2B, l2B, transpose_b=True)
            # sim = sim - tf.eye(tf.shape(sim)[0], tf.shape(sim)[1])

            l2B = tf.nn.l2_normalize(self.predictor.B, -1)
            sim = tf.matmul(l2B, l2B, transpose_b=True)
            # sim = sim - tf.eye(tf.shape(sim)[0], tf.shape(sim)[1])
            sim = sim - tf.eye(tf.shape(sim)[1], tf.shape(sim)[2], [tf.shape(sim)[0]])
            # reg_loss = tf.reduce_max(sim)
            reg_loss = tf.reduce_mean(tf.nn.relu(sim))

            # sim = tf.reduce_sum(S_list[:-1] * V_list, -1)
            # # reg_loss = tf.reduce_mean(sim)
            # mask = tf.cast(tf.greater(sim, 0.9), tf.float32)
            # reg_loss = tf.math.divide_no_nan(sim * mask, tf.reduce_sum(mask)) * 100.

        trainable_weights = self.predictor.trainable_weights
        grads = tape.gradient([meta_loss, lossRec, reg_loss], trainable_weights)
        # grads = tape.gradient([lossRec], trainable_weights)
        # grads = tape.gradient([meta_loss, lossRec], trainable_weights)
        self.opt2.apply_gradients(zip(grads, trainable_weights))
        self.reconstruction_loss.update_state(lossRec)
        self.query_loss_metric.update_state(meta_loss)
        self.entropy_metric.update_state(reg_loss)
        logs = {
            self.query_loss_metric.name: self.query_loss_metric.result(),
            self.reconstruction_loss.name: self.reconstruction_loss.result(),
            self.entropy_metric.name: self.entropy_metric.result(),
        }
        return logs

    def test_step(self, data):
        support, query = data
        support_image, support_label, support_label_global = support
        query_image, query_label, _ = query

        batch = tf.shape(support_image)[0]
        ways = tf.shape(support_image)[1]
        shots = tf.shape(support_image)[2]
        query_shots = tf.shape(query_image)[2]

        support_image_1shot = tf.reshape(support_image[:, :, :1, ...], [-1, *tf.unstack(tf.shape(support_image)[-3:])])
        support_image = tf.reshape(support_image, [-1, *tf.unstack(tf.shape(support_image)[-3:])])
        query_image = tf.reshape(query_image, [-1, *tf.unstack(tf.shape(query_image)[-3:])])
        training = False
        # S_1s = self(support_image_1shot, training=training, pool=False)
        # _, f_h, f_w, f_c = tf.unstack(tf.shape(S_1s))
        # S_1s = tf.reshape(S_1s, [batch, ways, -1, tf.shape(S_1s)[-1]])
        # S_1s_label = tf.repeat(support_label[:, :, :1, :], f_h * f_w, 2)
        S_1s = self(support_image_1shot, training=training, pool=True)
        S_1s = tf.reshape(S_1s, [batch, ways, -1, tf.shape(S_1s)[-1]])
        S_1s_label = support_label[:, :, :1, :]

        S = self(support_image, training=training, pool=True)
        Q = self(query_image, training=training)
        S = tf.reshape(S, [batch, ways, -1, tf.shape(S)[-1]])
        Q = tf.reshape(Q, [batch, ways, query_shots, tf.shape(Q)[-1]])

        test_pred = self.predictor.infer((S, support_label), Q, iterNum=self.iterNum, training=training)
        test_pred_max = test_pred[-1]
        acc_base_max = tf.keras.metrics.categorical_accuracy(
            tf.reshape(query_label, [batch, -1, tf.shape(query_label)[-1]]),
            tf.reshape(test_pred_max, [batch, -1, tf.shape(test_pred_max)[-1]]))
        acc_base_max = tf.reduce_mean(acc_base_max, -1)

        test_pred_1s = self.predictor.infer((S_1s, S_1s_label), Q, iterNum=self.iterNum_1s, training=training)

        test_pred_1s_max = test_pred_1s[-1]

        acc_one_shot_max = tf.keras.metrics.categorical_accuracy(
            tf.reshape(query_label, [batch, -1, tf.shape(query_label)[-1]]),
            tf.reshape(test_pred_1s_max, [batch, -1, tf.shape(test_pred_1s_max)[-1]]))
        acc_one_shot_max = tf.reduce_mean(acc_one_shot_max, -1)

        acc_base = tf.keras.metrics.categorical_accuracy(
            tf.reshape(query_label, [1, batch, -1, tf.shape(query_label)[-1]]),
            tf.reshape(test_pred, [self.iterNum + 1, batch, -1, tf.shape(test_pred)[-1]]))
        for index in range(self.iterNum + 1):
            self.mean_query_acc_group[index].update_state(tf.reduce_mean(acc_base, -1)[index])

        acc_one_shot = tf.keras.metrics.categorical_accuracy(
            tf.reshape(query_label, [1, batch, -1, tf.shape(query_label)[-1]]),
            tf.reshape(test_pred_1s, [self.iterNum_1s + 1, batch, -1, tf.shape(test_pred_1s)[-1]]))

        for index in range(self.iterNum_1s + 1):
            self.mean_query_acc_1s_group[index].update_state(tf.reduce_mean(acc_one_shot, -1)[index])

        self.mean_query_acc.update_state(tf.reduce_max(tf.reduce_mean(acc_base, -1), 0))
        self.mean_query_acc_1s.update_state(tf.reduce_max(tf.reduce_mean(acc_one_shot, -1), 0))
        self.mean_query_acc_base.update_state(acc_base_max)
        self.mean_query_acc_one_shot.update_state(acc_one_shot_max)

        logs = {
            self.mean_query_acc.name: self.mean_query_acc.result(),
            self.mean_query_acc_1s.name: self.mean_query_acc_1s.result(),
            self.mean_query_acc_base.name: self.mean_query_acc_base.result(),
            self.mean_query_acc_one_shot.name: self.mean_query_acc_one_shot.result(),
        }
        logs.update({m.name: m.result() for m in self.mean_query_acc_group})
        logs.update({m.name: m.result() for m in self.mean_query_acc_1s_group})
        return logs

    # def meta_train_step(self, data):
    #     ds_episode, ds_episode_embedding, ds_episode_embedding_blocks, ds_episode_embedding_cluster = data
    #     support, query = ds_episode
    #     support_image, S_label, support_global_label = support
    #     query_image, Q_label, query_global_label = query
    #     batch = tf.shape(support_image)[0]
    #     ways = tf.shape(support_image)[1]
    #     shots = tf.shape(support_image)[2]
    #     query_shots = tf.shape(query_image)[2]
    #     global_dim_shape = tf.shape(support_global_label)[-1]
    #
    #     training = True
    #     support_image = tf.reshape(support_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))
    #     query_image = tf.reshape(query_image, tf.concat([[-1], tf.shape(query_image)[-3:]], axis=0))
    #     support_global_label = tf.reshape(support_global_label, [-1, global_dim_shape])
    #
    #     teacher_S, teacher_Q = ds_episode_embedding
    #     teacher_Z = tf.concat([teacher_S, teacher_Q], 2)
    #     # teacher_Z = tf.reshape(teacher_Z, [-1, tf.shape(teacher_Z)[-1]])
    #     teacher_clusters, teacher_clusters_valid_mask, teacher_cluster_nums = ds_episode_embedding_cluster
    #     teacher_category_embedding_blocks, teacher_category_embedding_blocks_valid_mask, teacher_category_embedding_blocks_nums = ds_episode_embedding_blocks
    #
    #     Z_label = tf.concat([S_label, Q_label], 2)
    #     iterNum = self.iterNum
    #
    #     # @tf.function
    #     # def process_tensors(blocks, mask):
    #     #     valid_data_flat = tf.ragged.boolean_mask(tf.reshape(blocks,
    #     #                                                         [batch * ways, -1,
    #     #                                                          tf.shape(blocks)[-1]]),
    #     #                                              tf.reshape(
    #     #                                                  tf.cast(mask, tf.bool),
    #     #                                                  [batch * ways, -1]))
    #     #     valid_data_grouped = tf.RaggedTensor.from_row_lengths(valid_data_flat.flat_values,
    #     #                                                           valid_data_flat.row_lengths())
    #     #     steps = tf.maximum(tf.cast(valid_data_flat.row_lengths() / iterNum + 0.5, tf.int32), 1)
    #     #
    #     #     min_step = tf.cast(tf.reduce_min(valid_data_flat.row_lengths()), tf.int32)
    #     #     ProtoList = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    #     #     SampledZ = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    #     #
    #     #     for i in tf.range(tf.shape(steps)[0]):
    #     #         x = valid_data_grouped[i]  # 获取数据
    #     #         step = steps[i]  # 获取步骤
    #     #         Protos = tf.TensorArray(dtype=tf.float32, size=iterNum)
    #     #         for it in range(0, iterNum):
    #     #             Protos = Protos.write(it, tf.reduce_mean(x[:step * (it + 1)], axis=0))
    #     #         ProtoList = ProtoList.write(i, Protos.stack())
    #     #         indices = tf.random.shuffle(tf.range(tf.shape(x)[0]))[:min_step]
    #     #         SampledZ = SampledZ.write(i, tf.gather(x, indices))
    #     #     ProtoList = ProtoList.stack()
    #     #     ProtoList = tf.reshape(ProtoList, [batch, ways, iterNum, -1])
    #     #     SampledZ = SampledZ.stack()
    #     #     SampledZ = tf.reshape(SampledZ, [batch, ways, min_step, -1])
    #     #
    #     #     # 示例中没有 S_label 的定义，所以这里假设有一个相应的 S_label 张量
    #     #     SampledZ_label = tf.repeat(S_label[:, :, :1, :], min_step, axis=2)
    #     #     return ProtoList, SampledZ, SampledZ_label
    #     #
    #     # ProtoList, SampledZ, SampledZ_label = process_tensors(teacher_category_embedding_blocks,
    #     #                                                       teacher_category_embedding_blocks_valid_mask)
    #
    #     with tf.GradientTape() as tape:
    #
    #         S = self(support_image, training=training, pool=True)
    #         # _, f_h, f_w, f_c = tf.unstack(tf.shape(S))
    #         # S = tf.reshape(S, [batch, ways, -1, tf.shape(S)[-1]])
    #         # S_label = tf.repeat(S_label, f_h * f_w, 2)
    #
    #         Q = self(query_image, training=training)
    #         S = tf.reshape(S,
    #                        [batch, ways, -1, tf.shape(S)[-1]])
    #         Q = tf.reshape(Q,
    #                        [batch, ways, -1, tf.shape(Q)[-1]])
    #
    #         S = tf.nn.l2_normalize(S, -1)
    #         Q = tf.nn.l2_normalize(Q, -1)
    #         Z = tf.concat([S, Q], 2)
    #
    #         SShape = tf.shape(S)
    #         dims = SShape[-1]
    #
    #         S_list, P_list, P_ = self.predictor(S=S,
    #                                             S_label=S_label,
    #                                             training=training, iterNum=iterNum)
    #         Q_pred = self.predictor.clc(
    #             tf.repeat(tf.reshape(Q, [1, batch, -1, tf.shape(Q)[-1]]), iterNum + 1, 0),
    #             tf.reshape(P_list, [iterNum + 1, batch, ways, dims]))
    #
    #         meta_contrast_loss = tf.keras.losses.categorical_crossentropy(
    #             tf.repeat(tf.reshape(Q_label, [1, batch, -1, tf.shape(Q_label)[-1]]), iterNum + 1, 0),
    #             tf.reshape(Q_pred, [iterNum + 1, batch, -1, tf.shape(Q_pred)[-1]]))
    #
    #         meta_contrast_loss = tf.reduce_mean(meta_contrast_loss[1:], -1)
    #
    #         # sim = 1. + tf.losses.cosine_similarity(tf.reshape(ProtoList, [iterNum, batch, ways, 1, -1]),
    #         #                                        S_list[1:], -1)
    #         # recon_loss = tf.reduce_mean(sim[1:], 0)
    #         meta_loss = tf.reduce_mean(meta_contrast_loss)
    #         # lossRec = tf.reduce_mean(recon_loss)
    #
    #         l2B = tf.reshape(tf.nn.l2_normalize(self.predictor.B, -1),
    #                          [-1, tf.shape(self.predictor.B)[-1]])
    #         sim = tf.matmul(l2B, l2B, transpose_b=True)
    #         sim = sim - tf.eye(tf.shape(sim)[0], tf.shape(sim)[1])
    #         reg_loss = tf.reduce_mean(tf.nn.relu(sim))
    #
    #     trainable_vars = self.predictor.trainable_weights + self.encoder.trainable_weights
    #     # trainable_vars = self.predictor.trainable_weights
    #     # grads = tape.gradient([meta_loss, recon_loss, reg_loss], trainable_vars)
    #     grads = tape.gradient([meta_loss, reg_loss], trainable_vars)
    #     self.optimizer.apply_gradients(zip(grads, trainable_vars))
    #
    #     # self.reconstruction_loss.update_state(lossRec)
    #     self.query_loss_metric.update_state(meta_loss)
    #     self.entropy_metric.update_state(reg_loss)
    #     logs = {
    #         self.query_loss_metric.name: self.query_loss_metric.result(),
    #         self.reconstruction_loss.name: self.reconstruction_loss.result(),
    #         self.entropy_metric.name: self.entropy_metric.result(),
    #         # self.entropy_metric.name: self.entropy_metric.result(),
    #     }
    #     return logs

    def meta_train_step(self, data):
        support, query = data
        support_image, S_label, support_global_label = support
        all_query_image, all_query_label, all_query_global_label = query
        query_image, Q_label = all_query_image[:, :, :5, ...], all_query_label[:, :, :5, ...]
        all_image, all_label = tf.concat([support_image, all_query_image], 2), tf.concat([S_label, all_query_label], 2)
        batch = tf.shape(support_image)[0]
        ways = tf.shape(support_image)[1]
        shots = tf.shape(support_image)[2]
        query_shots = tf.shape(query_image)[2]

        global_dim_shape = tf.shape(support_global_label)[-1]
        training = True
        support_image = tf.reshape(support_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))
        query_image = tf.reshape(query_image, tf.concat([[-1], tf.shape(query_image)[-3:]], axis=0))
        support_global_label = tf.reshape(support_global_label, [-1, global_dim_shape])
        all_image = tf.reshape(all_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))
        Z = self(all_image, training=False)
        Z = tf.reshape(Z, [batch, ways, -1, tf.shape(Z)[-1]])
        Z = tf.nn.l2_normalize(Z, -1)
        iterNum = 5
        ProtoList = []
        step = tf.maximum((tf.shape(Z)[-2] - shots - query_shots) // iterNum, 1)
        for it in range(1, iterNum + 1):
            ProtoList.append(tf.reduce_mean(Z[:, :, :shots + query_shots + step * it, :], 2))
        ProtoList = tf.stack(ProtoList, 0)
        target = tf.reduce_mean(Z, 2)

        with tf.GradientTape() as tape:
            S = self(support_image, training=training, pool=False)
            # S = self.last_max_pooling(self.encoder(support_image,training=training))
            # _, f_h, f_w, f_c = tf.unstack(tf.shape(S))
            # S = tf.reshape(S, [batch, ways, -1, tf.shape(S)[-1]])
            # S_label = tf.repeat(S_label, f_h * f_w, 2)
            S = tf.reshape(S,
                           [batch, ways, -1, tf.shape(S)[-1]])
            S = tf.nn.l2_normalize(S, -1)
            Q = self(query_image, training=training)
            Q = tf.reshape(Q,
                           [batch, ways, -1, tf.shape(Q)[-1]])

            SShape = tf.shape(S)
            dims = SShape[-1]

            Q_list, _, _ = self.predictor(S=Q,
                                          S_label=Q_label,
                                          training=training, iterNum=iterNum)

            S_list, P_list, _ = self.predictor(S=S,
                                               S_label=S_label,
                                               training=training, iterNum=iterNum)

            Q_Pred = self.predictor.clc(tf.reshape(Q, [batch, -1, dims]),
                                        tf.reshape(P_list[-1], [batch, ways, dims]))
            meta_contrast_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    tf.reshape(Q_label, [batch, -1, tf.shape(Q_label)[-1]]),
                    tf.reshape(Q_Pred, [batch, -1, tf.shape(Q_Pred)[-1]])),
                -1)

            # meta_contrast_loss = tf.reduce_mean(meta_contrast_loss, 0)

            Z_sub = tf.concat([S_list, Q_list], -2)
            sim = 1. + tf.losses.cosine_similarity(tf.reshape(ProtoList, [iterNum, batch, ways, 1, -1]),
                                                   Z_sub[1:], -1)
            simTarget = 1. + tf.losses.cosine_similarity(tf.reshape(target, [1, batch, ways, 1, -1]),
                                                         Z_sub[1:], -1)
            simCmp = 1. + tf.losses.cosine_similarity(tf.reshape(target, [1, batch, ways, 1, -1]),
                                                      tf.reshape(ProtoList, [iterNum, batch, ways, 1, -1]), -1)
            mask = tf.cast(tf.less(simCmp, simTarget), tf.float32)
            sim = mask * sim
            recon_loss = tf.reduce_mean(sim, 0)
            meta_loss = tf.reduce_mean(meta_contrast_loss)
            lossRec = tf.reduce_mean(recon_loss)

            l2B = tf.reshape(tf.nn.l2_normalize(self.predictor.B, -1),
                             [-1, tf.shape(self.predictor.B)[-1]])
            sim = tf.matmul(l2B, l2B, transpose_b=True)
            sim = sim - tf.eye(tf.shape(sim)[0], tf.shape(sim)[1])
            reg_loss = tf.reduce_max(sim)
            # reg_loss = tf.reduce_mean(tf.nn.relu(sim))

        trainable_vars = self.predictor.trainable_weights + self.encoder.trainable_weights
        grads = tape.gradient([meta_loss, lossRec], trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.reconstruction_loss.update_state(lossRec)
        self.query_loss_metric.update_state(meta_loss)
        self.entropy_metric.update_state(reg_loss)
        logs = {
            self.query_loss_metric.name: self.query_loss_metric.result(),
            self.reconstruction_loss.name: self.reconstruction_loss.result(),
            self.entropy_metric.name: self.entropy_metric.result(),
            # self.entropy_metric.name: self.entropy_metric.result(),
        }
        return logs

    def E_S_Step(self, data):
        support, query = data
        support_image, support_label, support_global_label = support
        all_query_image, all_query_label, all_query_global_label = query
        query_image, query_label = all_query_image[:, :, :5, ...], all_query_label[:, :, :5, ...]
        rest_image, rest_label = all_query_image[:, :, 5:, ...], all_query_label[:, :, 5:, ...]
        batch = tf.shape(support_image)[0]
        ways = tf.shape(support_image)[1]
        shots = tf.shape(support_image)[2]
        query_shots = tf.shape(query_image)[2]
        rest_shots = tf.shape(rest_image)[2]

        global_dim_shape = tf.shape(support_global_label)[-1]
        training = True
        support_image = tf.reshape(support_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))
        query_image = tf.reshape(query_image, tf.concat([[-1], tf.shape(query_image)[-3:]], axis=0))
        support_global_label = tf.reshape(support_global_label, [-1, global_dim_shape])
        rest_image = tf.reshape(rest_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))
        rest_logits = self.mixGap(self.encoder(rest_image, training=False))
        rest_logits = tf.reshape(rest_logits,
                                 [batch, ways, -1, tf.shape(rest_logits)[-1]])
        with tf.GradientTape() as tape:
            support_features = self.encoder(support_image, training=training)
            _, f_h, f_w, f_c = tf.unstack(tf.shape(support_features))
            support_logits = self.mixGap(support_features)
            query_features = self.encoder(query_image, training=training)
            query_logits = self.mixGap(query_features)

            S = tf.reshape(support_logits,
                           [batch, ways, shots, tf.shape(support_logits)[-1]])
            query_logits = tf.reshape(query_logits,
                                      [batch, ways, query_shots, tf.shape(support_logits)[-1]])
            Z = tf.concat([S, rest_logits, query_logits], 2)
            # Z = tf.reshape(Z, [batch, ways * (shots + rest_shots + query_shots), -1])
            Z_label = tf.concat([support_label, rest_label, query_label], 2)
            Q = tf.concat([rest_logits, query_logits], 2)
            Q = tf.reshape(Q, [batch, ways * (rest_shots + query_shots), -1])
            Q_label = tf.concat([rest_label, query_label], 2)

            new_P, meta_loss, lossRec, entropy = self.predictor(S,
                                                                support_label,
                                                                reference=(tf.stop_gradient(Z), Z_label),
                                                                testset=(Q, Q_label),
                                                                training=training, iterNum=1)

            # new_P, meta_loss, lossRec, entropy = self.predictor.genRefinedProto((S, support_label), (Q, Q_label),
            #                                                                     training=True,
            #                                                                     iterNum=self.iterNum)

        trainable_vars = self.encoder.trainable_weights + self.predictor.trainable_weights
        # trainable_vars = self.encoder.trainable_weights
        # grads = tape.gradient([meta_loss, lossRec, entropy], trainable_vars)
        # grads = tape.gradient([meta_loss], trainable_vars)
        grads = tape.gradient([meta_loss, lossRec, entropy], trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        self.reconstruction_loss.update_state(lossRec)
        self.query_loss_metric.update_state(meta_loss)
        self.entropy_metric.update_state(entropy)
        # self.loss_metric.update_state(sim_loss)
        logs = {
            self.query_loss_metric.name: self.query_loss_metric.result(),
            self.reconstruction_loss.name: self.reconstruction_loss.result(),
            self.entropy_metric.name: self.entropy_metric.result(),
            # self.loss_metric.name: self.loss_metric.result(),
        }
        return logs

    def E_step(self, data):
        support, query = data
        support_image, support_label, support_global_label = support
        query_image, query_label, query_global_label = query

        batch = tf.shape(support_image)[0]
        ways = tf.shape(support_image)[1]
        shots = tf.shape(support_image)[2]
        query_shots = tf.shape(query_image)[2]

        global_dim_shape = tf.shape(support_global_label)[-1]
        training = True
        support_image = tf.reshape(support_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))
        query_image = tf.reshape(query_image, tf.concat([[-1], tf.shape(query_image)[-3:]], axis=0))
        support_global_label = tf.reshape(support_global_label, [-1, global_dim_shape])
        query_global_label = tf.reshape(query_global_label, [-1, global_dim_shape])

        with tf.GradientTape() as tape:
            support_features = self.encoder(support_image, training=training)
            _, f_h, f_w, f_c = tf.unstack(tf.shape(support_features))
            support_logits = self.mixGap(support_features)
            support_logits = tf.nn.l2_normalize(support_logits, -1)

            support_logits = tf.reshape(support_logits,
                                        [batch, ways, shots, tf.shape(support_logits)[-1]])
            support_mean, support_mean_label = random_sample_support(support_logits, support_label)
            # support_mean = tf.reduce_mean(support_logits, 2)

            query_features = self.encoder(query_image, training=training)
            query_logits = self.mixGap(query_features)
            query_logits = tf.nn.l2_normalize(query_logits, -1)

            logits_dim = tf.shape(support_mean)[-1]
            dim_shape = tf.shape(query_label)[-1]

            support_mean = tf.reshape(support_mean, [batch, ways, logits_dim])
            support_mean = tf.nn.l2_normalize(support_mean, -1)

            query_logits = tf.reshape(query_logits,
                                      [batch, -1, tf.shape(query_logits)[-1]])

            query_logits = tf.nn.l2_normalize(query_logits, -1)
            sim = tf.linalg.matmul(query_logits, support_mean, transpose_b=True)

            sim = tf.reshape(sim, [batch, ways, query_shots, -1])
            sim = tf.nn.softmax(sim * 20, -1)
            meta_contrast_loss = tf.keras.losses.categorical_crossentropy(
                tf.reshape(query_label, [-1, tf.shape(query_label)[-1]]),
                tf.reshape(sim, [-1, tf.shape(sim)[-1]]))
            meta_contrast_loss = tf.reduce_mean(meta_contrast_loss)

            avg_loss = meta_contrast_loss

        trainable_vars = self.encoder.trainable_weights
        grads = tape.gradient([avg_loss], trainable_vars)
        self.opt1.apply_gradients(zip(grads, trainable_vars))

        self.query_loss_metric.update_state(avg_loss)
        logs = {
            self.query_loss_metric.name: self.query_loss_metric.result(),
        }
        return logs

    def E_test_step(self, data):
        support, query = data
        support_image, support_label, _ = support
        query_image, query_label, _ = query

        batch = tf.shape(support_image)[0]
        ways = tf.shape(support_image)[1]
        shots = tf.shape(support_image)[2]
        query_shots = tf.shape(query_image)[2]

        support_image = tf.reshape(support_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))

        training = False

        support_features = self.encoder(support_image, training=training)
        _, f_h, f_w, f_c = tf.unstack(tf.shape(support_features))
        support_logits = self.mixGap(support_features)
        support_logits = tf.nn.l2_normalize(support_logits, -1)

        support_logits_base = tf.reshape(support_logits,
                                         [batch, ways, shots, tf.shape(support_logits)[-1]])
        support_logits_base = tf.nn.l2_normalize(support_logits_base, -1)
        x_mean_base = tf.reduce_mean(support_logits_base, 2)
        x_mean_one_shot = support_logits_base[..., 0, :]

        new_shape = tf.concat([[-1], tf.shape(query_image)[-3:]], axis=0)
        query_image = tf.reshape(query_image, new_shape)

        query_features = self.encoder(query_image, training=training)
        query_logits = self.mixGap(query_features)
        query_logits = tf.nn.l2_normalize(query_logits, -1)

        logits_dim = tf.shape(x_mean_base)[-1]
        dim_shape = tf.shape(query_label)[-1]

        support_mean_base = tf.reshape(x_mean_base, [batch, ways, logits_dim])
        support_mean_base = tf.nn.l2_normalize(support_mean_base, -1)
        reshape_query_logits_base = tf.reshape(query_logits, [batch, ways * query_shots, logits_dim])
        reshape_query_logits_base = tf.nn.l2_normalize(reshape_query_logits_base, -1)
        sim = tf.linalg.matmul(reshape_query_logits_base, support_mean_base, transpose_b=True)

        sim = tf.reshape(sim, [batch, ways, query_shots, -1])

        dist_one_shot = tf.linalg.matmul(reshape_query_logits_base, x_mean_one_shot, transpose_b=True)
        acc_one_shot = tf.keras.metrics.categorical_accuracy(tf.reshape(query_label, [batch, -1, dim_shape]),
                                                             dist_one_shot)
        acc_one_shot = tf.reduce_mean(acc_one_shot, -1)

        intra_dist = tf.math.divide_no_nan(tf.reduce_sum((-sim + 1.) * query_label, -1),
                                           tf.reduce_sum(query_label, -1))
        intra_dist = tf.reduce_mean(intra_dist)
        inter_dist = tf.math.divide_no_nan(tf.reduce_sum((-sim + 1.) * (1. - query_label), -1),
                                           tf.reduce_sum((1. - query_label), -1))
        inter_dist = tf.reduce_mean(inter_dist)
        sim = tf.nn.softmax(sim * 20, -1)
        entropy = tf.reduce_mean(tf.reduce_sum(-sim * tf.math.log(sim), -1))
        InfoNCE = tf.reduce_mean(-tf.math.log(tf.reduce_sum(sim * query_label, -1)))
        meta_contrast_loss = tf.keras.losses.categorical_crossentropy(
            tf.reshape(query_label, [-1, tf.shape(query_label)[-1]]),
            tf.reshape(sim, [-1, tf.shape(sim)[-1]]))
        meta_contrast_loss = tf.reduce_mean(meta_contrast_loss)

        acc_base = tf.keras.metrics.categorical_accuracy(tf.reshape(query_label, [batch, -1, dim_shape]),
                                                         tf.reshape(sim, [batch, -1, tf.shape(sim)[-1]]))
        acc_base = tf.reduce_mean(acc_base, -1)
        self.mean_query_acc_base.update_state(acc_base)
        self.mean_query_acc_one_shot.update_state(acc_one_shot)
        self.query_loss_metric.update_state(meta_contrast_loss)
        self.inter_metric.update_state(inter_dist)
        self.intra_metric.update_state(intra_dist)
        self.entropy_metric.update_state(entropy)
        self.InfoNCE_metric.update_state(InfoNCE)
        logs = {
            self.mean_query_acc_base.name: self.mean_query_acc_base.result(),
            self.mean_query_acc_one_shot.name: self.mean_query_acc_one_shot.result(),
            self.query_loss_metric.name: self.query_loss_metric.result(),
            self.intra_metric.name: self.intra_metric.result(),
            self.inter_metric.name: self.inter_metric.result(),
            self.entropy_metric.name: self.entropy_metric.result(),
            self.InfoNCE_metric.name: self.InfoNCE_metric.result(),
        }
        return logs

    def get_category_embedding(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "train"), exist_ok=True);
        os.makedirs(os.path.join(save_dir, "test"), exist_ok=True);
        os.makedirs(os.path.join(save_dir, "val"), exist_ok=True);
        training = False
        from tqdm import tqdm
        from multiprocessing.dummy import Pool as ThreadPool

        def process_save(data):
            z, label, path = data
            new_path = path.decode().replace(self.data_dir_path, save_dir) + ".z"
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            tf.io.write_file(new_path, tf.io.serialize_tensor(z))

        for data in tqdm(self.all_train_ds):
            x, y, pathList = data
            features = self.encoder(x, training=training)
            logits = self.mixGap(features)
            pool = ThreadPool(24)  # 创建10个容量的线程池并发执行
            pool.imap_unordered(process_save,
                                zip(logits.numpy(),
                                    tf.argmax(y, -1).numpy(), pathList.numpy()))

            pool.close()
            pool.join()

        print("done..!")

    def generate_anchors(self, data_dir_path="/data/giraffe/0_FSL/SE_FS_embedding", anchor_num=8):
        dataloaderEmb = DataLoaderEmbedding(data_dir_path=data_dir_path)
        self.embedding_achor_dict, self.embeddingAnchors, self.embeddingLabels = dataloaderEmb.generate_anchors("train",
                                                                                                                anchor_num=anchor_num)

    def loadEmbeddingAnchors(self, anchor_num=8,
                             embedding_path="/data/di/groupcs/dataset/classification/20220506_FSL_data/data/processed_images_224",
                             multiView_embedding_path=[],
                             cache=True):
        anchor_num = 10
        if cache is True:
            print("using cache ...")
            try:
                dataloaderEmb = DataLoaderEmbedding(data_dir_path=embedding_path,
                                                    multiView_embedding_path=multiView_embedding_path)
                if os.path.exists(os.path.join(embedding_path, "embeddingAnchors.bin")) and os.path.exists(
                        os.path.join(embedding_path, "embeddingLabels.bin")):
                    self.embeddingAnchors = tf.io.parse_tensor(
                        tf.io.read_file(os.path.join(embedding_path, "embeddingAnchors.bin")),
                        tf.float32)
                    self.embeddingLabels = tf.io.parse_tensor(
                        tf.io.read_file(os.path.join(embedding_path, "embeddingLabels.bin")),
                        tf.float32)
                    print("~~~~!!!!loading embeddingAnchors to {}".format(
                        os.path.join(embedding_path, "embeddingAnchors.bin")))
                else:
                    self.embedding_achor_dict, self.embeddingAnchors, self.embeddingLabels = dataloaderEmb.generate_anchors(
                        "train", anchor_num=anchor_num)
                    tf.io.write_file(os.path.join(embedding_path, "embeddingAnchors.bin"),
                                     tf.io.serialize_tensor(self.embeddingAnchors))
                    tf.io.write_file(os.path.join(embedding_path, "embeddingLabels.bin"),
                                     tf.io.serialize_tensor(self.embeddingLabels))
                    print("~~~~!!!!saving embeddingAnchors to {}".format(
                        os.path.join(embedding_path, "embeddingAnchors.bin")))
            except:
                traceback.print_exc()
                self.get_category_embedding(save_dir=embedding_path)
                dataloaderEmb = DataLoaderEmbedding(data_dir_path=embedding_path,
                                                    multiView_embedding_path=multiView_embedding_path)
                self.embedding_achor_dict, self.embeddingAnchors, self.embeddingLabels = dataloaderEmb.generate_anchors(
                    "train",
                    anchor_num=anchor_num)
                tf.io.write_file(os.path.join(embedding_path, "embeddingAnchors.bin"),
                                 tf.io.serialize_tensor(self.embeddingAnchors))
                tf.io.write_file(os.path.join(embedding_path, "embeddingLabels.bin"),
                                 tf.io.serialize_tensor(self.embeddingLabels))

        else:
            print("no cache ...", embedding_path)
            self.get_category_embedding(save_dir=embedding_path)
            dataloaderEmb = DataLoaderEmbedding(data_dir_path=embedding_path,
                                                multiView_embedding_path=multiView_embedding_path)
            self.embedding_achor_dict, self.embeddingAnchors, self.embeddingLabels = dataloaderEmb.generate_anchors(
                "train",
                anchor_num=anchor_num)
            tf.io.write_file(os.path.join(embedding_path, "embeddingAnchors.bin"),
                             tf.io.serialize_tensor(self.embeddingAnchors))
            tf.io.write_file(os.path.join(embedding_path, "embeddingLabels.bin"),
                             tf.io.serialize_tensor(self.embeddingLabels))

        # self.predictor.updateAnchors(self.embeddingAnchors)
        # print("self.embeddingAnchors.shape:", self.embeddingAnchors.shape)
        self.embeddingAnchors = tf.reshape(self.embeddingAnchors, [-1, anchor_num, self.embeddingAnchors.shape[1]])
        print("origin embeddingAnchors.shape:", self.embeddingAnchors.shape)
        self.embeddingAnchors = tf.reduce_mean(self.embeddingAnchors, 1)
        print("reduce_mean embeddingAnchors.shape:", self.embeddingAnchors.shape)
        try:
            if self.anchor_init_flag is not True:
                self.predictor.updateAnchors(self.embeddingAnchors)
                print("self.embeddingAnchors.shape:", self.embeddingAnchors.shape)
        except:
            traceback.print_exc()
            self.anchor_init_flag = True
            self.predictor.updateAnchors(self.embeddingAnchors)
            print("self.embeddingAnchors.shape:", self.embeddingAnchors.shape)

        return dataloaderEmb

    def fine_tune(self, num_base_embbeding=5, num_expresive=20, lr=0.001,
                  anchor_num=8,
                  embedding_path="/data/di/groupcs/dataset/classification/20220506_FSL_data/data/processed_images_224",
                  multiView_embedding_path=[],
                  episode_num=1200,
                  total_epoch=100,
                  validation_freq=1, cache=True, debug=False):
        dataloaderEmb = self.loadEmbeddingAnchors(anchor_num, embedding_path, multiView_embedding_path, cache)

        if debug:
            train_batch = 1
            episode_num = 120
            test_train_shot = 50
        else:
            train_batch = 60
            test_train_shot = int(120 * 3)

        steps_per_epoch = episode_num // train_batch
        train_epoch = min(total_epoch, 1)
        total_epoch = total_epoch
        for it_try in range(test_train_shot):
            try:
                print("test_train_shot : ", test_train_shot - it_try)
                meta_train_ds, meta_train_name_projector = dataloaderEmb.get_dataset(phase='train', way_num=5,
                                                                                     shot_num=5,
                                                                                     episode_test_sample_num=test_train_shot - it_try,
                                                                                     episode_num=episode_num,
                                                                                     batch=train_batch,
                                                                                     epochs=train_epoch
                                                                                     ,
                                                                                     cache_file="{}_cache_file".format(
                                                                                         embedding_path))
                break
            except:
                # traceback.print_exc()
                continue
        print("test_train_shot : ", test_train_shot - it_try)

        # cache_file = "{}_cache_file".format(embedding_path)
        # print("cache_file {}".format(cache_file))
        # if os.path.exists(cache_file):
        #     print("cache_file {} is exist, remove it: ".format(cache_file))
        #     os.remove(cache_file)
        # meta_train_ds = meta_train_ds.cache(cache_file)

        # scheduled_lrs = WarmUpStep(
        #     learning_rate_base=lr,
        #     warmup_learning_rate=0.000001,
        #     warmup_steps=steps_per_epoch * 3,
        # )

        scheduled_lrs = WarmUpCosine(
            learning_rate_base=lr,
            total_steps=total_epoch * steps_per_epoch,
            warmup_learning_rate=0.000001,
            warmup_steps=steps_per_epoch * 1,
        )
        self.opt2 = tf.optimizers.Adam(learning_rate=scheduled_lrs, name="opt2")
        # self.opt2 = Adan(learning_rate=lr, name="opt2")
        # self.opt2 = tf.optimizers.Adam(learning_rate=lr, name="opt2")
        # self.opt2 = tf.keras.optimizers.SGD(scheduled_lrs, momentum=0.9, nesterov=True, name="opt2")
        # self.opt2 = tf.keras.optimizers.SGD(scheduled_lrs, momentum=0.9, nesterov=False, name="opt2")
        self.compile(self.opt2)
        self.train_step = self.S_step
        if debug:
            from tqdm import tqdm
            for data in tqdm(meta_train_ds):
                print(self.train_step(data))

            for data in tqdm(self.meta_test_ds):
                print(
                    self.test_step(data))

        monitor_name = "val_{}".format("acc")
        monitor_cmp = "max"
        monitor_name = "val_mq_acc_{}".format(self.iterNum)
        monitor_cmp = "max"
        cur_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = self.name
        log_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_base_path = os.path.join(ckpt_base_path, "{}_ckpts/".format(name))
        print(log_base_path, ckpt_base_path)
        filepath = os.path.join(embedding_path,
                                "fine_tune_{}_best.h5".format(data_name))
        os.makedirs(embedding_path, exist_ok=True)

        tensorboard_save = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(embedding_path, '{}_finetune_logs/{}'.format(name, cur_date)),
            profile_batch=0, )
        checkpoint_save = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                             verbose=1, monitor=monitor_name,
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode=monitor_cmp)
        # self.fit(meta_train_ds.repeat(), epochs=total_epoch,
        #          steps_per_epoch=steps_per_epoch)
        callbacks = [
            checkpoint_save,
            tensorboard_save,
            tf.keras.callbacks.EarlyStopping(monitor_name, 0, 10, mode=monitor_cmp),
            PrintCallback()
        ]
        # self.evaluate(self.meta_test_ds)
        self.fit(meta_train_ds.repeat(), epochs=total_epoch,
                 steps_per_epoch=steps_per_epoch,
                 validation_data=self.meta_test_ds,
                 callbacks=callbacks, initial_epoch=0,
                 validation_freq=validation_freq)

        if filepath is not None:
            loadStatu = model.load_weights(filepath, by_name=True, skip_mismatch=True)
            print("finetune done. loading the best model:{},\n loadStatu {}".format(filepath, loadStatu))

    def show(self):
        def transpose_and_reshape(x):
            b, way, s, h, w, c = tf.unstack(tf.shape(x))
            x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
            x = tf.reshape(x, [b, way * h, s * w, c])
            return x

        from tqdm import tqdm
        for data in tqdm(self.meta_test_ds):
            support, query = data
            support_image, support_label, support_label_global = support
            query_image, query_label, _ = query

            batch = tf.shape(support_image)[0]
            ways = tf.shape(support_image)[1]
            shots = tf.shape(support_image)[2]
            query_shots = tf.shape(query_image)[2]

            support_image_1shot = tf.reshape(support_image[:, :, :1, ...],
                                             [-1, *tf.unstack(tf.shape(support_image)[-3:])])
            support_image = tf.reshape(support_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))
            query_image = tf.reshape(query_image, tf.concat([[-1], tf.shape(query_image)[-3:]], 0))

            training = False

            support_features_1s = self.encoder(support_image_1shot, training=training)
            S_1s = self.mixGap(support_features_1s)

            support_features_1s = tf.reshape(support_features_1s, [batch, ways, -1, tf.shape(support_features_1s)[-1]])
            S_1s = tf.reshape(S_1s, [batch, ways, -1, tf.shape(S_1s)[-1]])
            S_1s_label = support_label[:, :, :1, :]

            support_features = self.encoder(support_image, training=training)
            _, f_h, f_w, f_c = tf.unstack(tf.shape(support_features))
            support_logits = self.mixGap(support_features)
            support_logits = tf.nn.l2_normalize(support_logits, -1)
            query_features = self.encoder(query_image, training=training)
            query_logits = self.mixGap(query_features)
            query_logits = tf.nn.l2_normalize(query_logits, -1)

            S = tf.reshape(support_logits, tf.concat([[batch, ways, shots], tf.shape(support_logits)[-1:]], 0))
            Q = tf.reshape(query_logits, tf.concat([[batch, ways, query_shots], tf.shape(query_logits)[-1:]], 0))
            Z = tf.concat([S, Q], 2)
            Z_1s = tf.concat([S_1s, Q], 2)
            Z_label = tf.concat([support_label, query_label], 2)
            Z_1s_label = tf.concat([S_1s_label, query_label], 2)

            S_list, P = self.predictor(S=S, S_label=support_label, training=training, iterNum=self.iterNum)
            S_1s_list, P_1s = self.predictor(S_1s, S_1s_label, iterNum=self.iterNum_1s,
                                             training=training)

            sim = (1 - tf.losses.cosine_similarity(tf.reshape(support_features, [1, batch, ways, shots, f_h, f_w, f_c]),
                                                   tf.reshape(P, [self.iterNum + 1, batch, ways, 1, 1, 1, f_c]),
                                                   -1)) / 2.
            # sim = tf.clip_by_value(
            #     - tf.losses.cosine_similarity(tf.reshape(support_features_1s, [1, batch, ways, 1, f_h, f_w, f_c]),
            #                                   tf.reshape(P_1s, [self.iterNum + 1, batch, ways, 1, 1, 1, f_c]),
            #                                   -1), 0., 1.)
            sim = tf.transpose(sim, [1, 2, 3, 4, 5, 0])
            sim = transpose_and_reshape(sim)
            support_image = transpose_and_reshape(
                tf.reshape(support_image, [batch, ways, shots, *support_image.shape[-3:]]))

            for image, origin_s_attention in \
                    zip(support_image,
                        sim):
                image = (image[..., ::-1] * 255).numpy().astype(np.uint8)
                origin_s_attention_list = [image]
                for index in range(0, self.iterNum + 1, self.iterNum):
                    w = tf.image.resize(origin_s_attention[..., index:index + 1] * 255,
                                        image.shape[-3:-1],
                                        method='bilinear').numpy().astype(np.uint8)
                    w = cv2.applyColorMap(w, cv2.COLORMAP_JET)
                    w = cv2.addWeighted(image, 0.5, w, 0.5, 0)
                    origin_s_attention_list.append(w)
                show_image = cv2.hconcat(origin_s_attention_list)
                cv2.imshow("image", show_image)
                cv2.waitKey(0)
            # continue
            P = tf.transpose(P, [1, 2, 0, 3])
            P_1s = tf.transpose(P_1s, [1, 2, 0, 3])

            P = tf.reshape(P, [-1, tf.shape(P)[-1]])
            # P_1s = tf.reshape(P_1s, [-1, tf.shape(P_1s)[-1]])

            S = tf.reshape(S, [-1, tf.shape(S)[-1]])
            S_1s = tf.reshape(S_1s, [-1, tf.shape(S_1s)[-1]])
            Q = tf.reshape(Q, [-1, tf.shape(Q)[-1]])
            s_label = tf.reshape(tf.argmax(support_label, -1), [-1, 1]).numpy()
            s_label_1s = tf.reshape(tf.argmax(S_1s_label, -1), [-1, 1]).numpy()
            q_label = tf.reshape(tf.argmax(query_label, -1), [-1, 1]).numpy()
            p_labels = tf.reshape(tf.argmax(S_1s_label, -1), [-1, 1]).numpy()

            logits = tf.concat([S, Q], 0)
            all_lables = []
            all_lables.extend(["S{}".format(l) for l in s_label])
            all_lables.extend(["Q{}".format(l) for l in q_label])

            for l in p_labels:
                all_lables.extend(["P{}".format(l)])
                for iter in range(self.iterNum):
                    all_lables.extend(["P{}_{}".format(l, iter)])

            # self.show_TSNE(logits, all_lables)

            logits_1s = tf.concat([Q], 0)
            # logits_1s = tf.concat([Q, P_1s], 0)
            all_lables = []
            # all_lables.extend(["S{}".format(l) for l in s_label_1s])
            all_lables.extend(["Q{}".format(l) for l in q_label])

            path_logits = tf.concat([P_1s[...,:2,:],P_1s[...,self.iterNum_1s//2:self.iterNum_1s//2+1,:],P_1s[...,-1:,:]],-2)
            path_labels = []
            for l in p_labels:
                path_labels.extend(["P{}".format(l)])
                # for iter in range(self.iterNum_1s):
                for iter in range(3):
                    path_labels.extend(["P{}_{}".format(l, iter)])

            self.show_TSNE(logits_1s, all_lables, path=(path_logits, path_labels))

    def show_TSNE(self, data_input, labels, path=None, perplexity=250, info=""):
        from sklearn import manifold

        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import time
        '''t-SNE'''
        print("start t-SNE...", info)
        t0 = time.time()
        n_components = 2
        tsne = manifold.TSNE(n_components=n_components, init='pca', metric="cosine", random_state=0,
                             perplexity=perplexity)
        # tsne = TSNE(n_components=n_components, perplexity=200, n_iter=1000, initialization="pca",metric="cosine")
        # tsne = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine')

        data = tf.reshape(data_input, [-1, tf.shape(data_input)[-1]]).numpy()
        if path is not None:
            path, path_label = path
            path = tf.reshape(path, [-1, tf.shape(path)[-1]])
            data = np.concatenate([data, path], 0)
            labels.extend(path_label)
        data = tsne.fit_transform(data)

        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        t1 = time.time()
        print("t-SNE: %.2g sec" % (t1 - t0))
        # plt.figure(figsize=(20, 4))
        plt.figure(figsize=(8, 8))
        # plt.title("t-SNE (%.2g sec)" % (t1 - t0))
        label_dict = {v.split("_")[0][1:]: index for index, v in enumerate(sorted(list(set(labels))))}
        if len(label_dict) <= 5:
            colors = {label_dict[k]: v for k, v in zip(label_dict.keys(), ["red",  "green", "blue", "black", "orange"])}
        else:
            colors = list(mcolors.XKCD_COLORS.keys())
            random.shuffle(colors)
        ax = plt.gca()
        pathX = []
        pathY = []
        for i in range(data.shape[0]):
            text = labels[i]
            color = colors[label_dict[text.split("_")[0][1:]]]
            if "P" in text:

                if i > 1 and ((len(text.split("_")) == 2 and len(labels[i - 1].split("_")) == 1) or
                              (len(text.split("_")) == 2 and len(labels[i - 1].split("_")) == 2)):
                    pathX.append(data[i, 0])
                    pathY.append(data[i, 1])
                    # plt.plot(data[i, 0], data[i, 1], '*', color=color, markersize=20)

                    # if int(text.split("_")[1]) >= 1:
                    #     continue
                    #
                    # arrow = patches.FancyArrowPatch((data[i, 0], data[i, 1]), (data[i - 1, 0], data[i - 1, 1]),
                    #                                 arrowstyle='->', mutation_scale=20, color='red')
                    # ax.add_patch(arrow)
                    # plt.annotate('', xy=(data[i, 0], data[i, 1]), xytext=(data[i - 1, 0], data[i - 1, 1]),
                    # arrowprops=dict(facecolor='gray', shrink=0.2, linestyle='--'),
                    # arrowprops=dict(facecolor='none', edgecolor='black', shrink=0.3, linestyle='--',
                    #                 linewidth=1.5)
                    # arrowprops=dict(facecolor='black', edgecolor='black',
                    #                 shrink=0.2, headwidth=10, headlength=10, width=2)
                    # )

                else:
                    pathX.clear()
                    pathY.clear()
                    plt.scatter(data[i, 0], data[i, 1], marker='o', color=color, edgecolors=color, s=100)

                if len(pathX) > 1 and (
                        i + 1 >= data.shape[0] or i + 1 < data.shape[0] and len(labels[i + 1].split("_")) != 2):
                    # plt.quiver(pathX[:-1], pathY[:-1], np.diff(pathX), np.diff(pathY), angles='xy', scale_units='xy',
                    #            scale=1, color=color)
                    plt.quiver(
                        pathX[:-1], pathY[:-1],
                        np.diff(pathX), np.diff(pathY),
                        angles='xy', scale_units='xy', scale=1,
                        color="black",
                        width=0.005,  # 箭头宽度
                        headwidth=3,  # 箭头头部宽度
                        headlength=4,  # 箭头头部长度
                        headaxislength=4.5,  # 箭头头部与箭身相交的轴长
                        alpha=0.8,  # 箭头透明度
                        linestyle='--'  # 箭头线的样式
                    )
                    plt.plot(data[i, 0], data[i, 1], '*', color=color, markersize=20)
                    pass

            else:
                if "S" in text:
                    plt.scatter(data[i, 0], data[i, 1], marker='o', color=color, edgecolors=color, s=100)
                else:
                    plt.scatter(data[i, 0], data[i, 1], marker='o', color='none', edgecolors=color, s=100)

        # for i in range(data.shape[0]):
        #     text = labels[i]
        #     color = colors[-label_dict[text.split("_")[-1][1:]]]
        #     if "P" in text:
        #         plt.plot(data[i, 0], data[i, 1], '*', color=color, markersize=20)
        #     elif "n" in text:
        #         plt.plot(data[i, 0], data[i, 1], '^', color=color, markersize=20)
        #     elif "Z" in text:
        #         plt.plot(data[i, 0], data[i, 1], '*', color=color, markersize=8)
        #     elif "z" in text:
        #         plt.scatter(data[i, 0], data[i, 1], marker='^', color=color, edgecolors=color, s=40)
        #     elif "q" in text:
        #         plt.scatter(data[i, 0], data[i, 1], marker='o', color='none', edgecolors=color, s=40)
        #     else:
        #         # plt.text(data[i, 0], data[i, 1], text,
        #         #          color=color,
        #         #          fontdict={'weight': 'bold', 'size': 9})
        #         if "S" in text:
        #             # plt.plot(data[i, 0], data[i, 1], 'o', color=color, markersize=15)
        #             plt.scatter(data[i, 0], data[i, 1], marker='o', color=color, edgecolors=color, s=100)
        #         else:
        #             plt.scatter(data[i, 0], data[i, 1], marker='o', color='none', edgecolors=color, s=100)

        plt.xticks([])
        plt.yticks([])
        # plt.legend()
        # plt.axis('off')
        plt.tight_layout()
        plt.savefig("showRefinementPath.png", dpi=400)
        plt.show()

    def train_step_meta(self, data):
        support, query = data
        support_image, support_label, support_label_global = support
        query_image, query_label, _ = query

        batch = tf.shape(support_image)[0]
        ways = tf.shape(support_image)[1]
        shots = tf.shape(support_image)[2]
        query_shots = tf.shape(query_image)[2]

        support_image = tf.reshape(support_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))

        training = True

        with tf.GradientTape() as tape:
            support_features = self.encoder(support_image, training=training)
            _, f_h, f_w, f_c = tf.unstack(tf.shape(support_features))
            support_logits = self.mixGap(support_features)

            support_z = tf.reshape(support_logits, [batch, ways, shots, -1])
            support_z_proto = tf.reduce_mean(support_z, 2, keepdims=True)

            with tape.stop_recording():
                reference_label = tf.nn.softmax(tf.matmul(
                    tf.nn.l2_normalize(tf.reshape(support_z_proto, [batch * ways, -1]), -1),
                    tf.nn.l2_normalize(self.embeddingAnchors, -1), transpose_b=True), -1)
                reference_label = tf.reshape(reference_label,
                                             tf.concat([[batch, ways], tf.shape(reference_label)[-1:]], 0))

                z_hat = self.expresive(tf.reshape(support_z, [batch, ways, shots, -1]), reference_label,
                                       training=training)
                z_hat_ = self.expresive_1s(tf.reshape(support_z_proto, [batch, ways, 1, -1]), reference_label,
                                           training=training)
                # z_hat_1s = self.expresive_1s(tf.reshape(support_z_1s, [batch, ways, 1, -1]), reference_label_1s,
                #                              training=training)

                # tf.print(z_hat.shape, z_hat_1s.shape)
                z_hat = tf.reshape(z_hat, [batch, ways, -1, tf.shape(z_hat)[-1]])
                z_hat = tf.nn.l2_normalize(z_hat, -1)
                z_hat_ = tf.reshape(z_hat_, [batch, ways, -1, tf.shape(z_hat_)[-1]])
                z_hat_ = tf.nn.l2_normalize(z_hat_, -1)

                # z_hat_1s = tf.reshape(z_hat_1s, [batch, ways, -1, tf.shape(z_hat_1s)[-1]])
                # z_hat_1s = tf.nn.l2_normalize(z_hat_1s, -1)

            support_z = tf.nn.l2_normalize(support_z, -1)
            support_z_1s = tf.nn.l2_normalize(support_z_proto, -1)
            #
            support_z = tf.reshape(support_z,
                                   [batch, ways, shots, tf.shape(support_z)[-1]])
            support_z_1s = tf.reshape(support_z_1s,
                                      [batch, ways, 1, tf.shape(support_z_1s)[-1]])
            x_mean_base = tf.concat([support_z, z_hat], 2)
            # x_mean_base = tf.concat([support_z], 2)
            # x_mean_base = tf.concat([z_hat], 2)
            # x_mean_base = z_hat
            x_mean_base = tf.reduce_mean(x_mean_base, 2)
            x_mean_base = tf.nn.l2_normalize(x_mean_base, -1)

            x_mean_one_shot = tf.concat([support_z_1s, z_hat_], 2)
            # x_mean_one_shot = tf.concat([z_hat_1s], 2)
            # x_mean_one_shot = support_z
            x_mean_one_shot = tf.reduce_mean(x_mean_one_shot, 2)
            x_mean_one_shot = tf.nn.l2_normalize(x_mean_one_shot, -1)

            new_shape = tf.concat([[-1], tf.shape(query_image)[-3:]], axis=0)
            query_image = tf.reshape(query_image, new_shape)

            query_features = self.encoder(query_image, training=training)
            query_logits = self.mixGap(query_features)
            query_logits = tf.nn.l2_normalize(query_logits, -1)

            logits_dim = tf.shape(x_mean_base)[-1]
            dim_shape = tf.shape(query_label)[-1]

            reshape_query_logits_base = tf.reshape(query_logits, [batch, ways * query_shots, logits_dim])
            reshape_query_logits_base = tf.nn.l2_normalize(reshape_query_logits_base, -1)
            sim = tf.linalg.matmul(reshape_query_logits_base, x_mean_base, transpose_b=True)
            sim = tf.reshape(sim, [batch, ways, query_shots, -1])
            sim = tf.nn.softmax(sim * 20, -1)
            meta_contrast_loss = tf.keras.losses.categorical_crossentropy(
                tf.reshape(query_label, [-1, tf.shape(query_label)[-1]]),
                tf.reshape(sim, [-1, tf.shape(sim)[-1]]))

            sim_one_shot = tf.linalg.matmul(reshape_query_logits_base, x_mean_one_shot, transpose_b=True)
            sim_one_shot = tf.reshape(sim_one_shot, [batch, ways, query_shots, -1])
            sim_one_shot = tf.nn.softmax(sim_one_shot * 20, -1)

            meta_contrast_loss += tf.keras.losses.categorical_crossentropy(
                tf.reshape(query_label, [-1, tf.shape(query_label)[-1]]),
                tf.reshape(sim_one_shot, [-1, tf.shape(sim_one_shot)[-1]]))

        trainable_vars = self.encoder.trainable_weights
        # trainable_vars = self.encoder.trainable_weights + self.expresive.trainable_weights + self.expresive_1s.trainable_weights
        grads = tape.gradient([meta_contrast_loss], trainable_vars)
        self.opt1.apply_gradients(zip(grads, trainable_vars))

        self.query_loss_metric.update_state(meta_contrast_loss)
        logs = {
            self.query_loss_metric.name: self.query_loss_metric.result(),
        }
        return logs

    def E_train(self, lr=0.001, weights=None, total_epoch=50):
        if weights is not None:
            self.load_weights(weights, by_name=True, skip_mismatch=True)

        scheduled_lrs = WarmUpCosine(
            learning_rate_base=lr,
            total_steps=total_epoch * self.steps_per_epoch,
            warmup_learning_rate=0.0,
            warmup_steps=self.steps_per_epoch * 3,
        )
        self.opt1 = tf.keras.optimizers.SGD(scheduled_lrs, momentum=0.9, nesterov=True, name="opt2")
        self.compile(self.opt1)
        self.train_step = self.E_step
        self.test_step = self.E_test_step
        self.test_step = self.test_step_meta

        # for data in meta_train_ds:
        #     self.train_step(data)
        # for data in meta_test_ds:
        #     self.test_step_meta(data)
        monitor_name = "val_{}".format("acc")
        monitor_cmp = "max"
        monitor_name = "val_mq_acc_base"
        monitor_cmp = "max"
        cur_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = self.name
        log_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_base_path = os.path.join(ckpt_base_path, "{}_ckpts/".format(name))
        print(log_base_path, ckpt_base_path)
        filepath = os.path.join(ckpt_base_path,
                                "model_e{}-l {}.h5".format("{epoch:03d}",
                                                           "{" + "{}:.5f".format(monitor_name) + "}"))
        os.makedirs(ckpt_base_path, exist_ok=True)

        tensorboard_save = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_base_path, '{}_logs/{}'.format(name, cur_date)),
            profile_batch=0, )
        checkpoint_save = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                             verbose=1, monitor=monitor_name,
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode=monitor_cmp)

        callbacks = [
            checkpoint_save,
            tensorboard_save,
        ]
        self.evaluate(self.meta_test_ds)
        self.fit(self.meta_train_ds.repeat(), epochs=total_epoch,
                 steps_per_epoch=self.steps_per_epoch,
                 validation_data=self.meta_test_ds,
                 callbacks=callbacks, initial_epoch=0)

    def setDataset(self, data_dir_path, train_episode_num=1200, train_batch=4, ways=5, shots=5,
                   mix_up=False,
                   augment=True,
                   test_episode_num=600,
                   test_train_shot=5,
                   test_batch=8, test_shots=15, train_epoch=20, cross_domain=False,
                   embedding_dir_path=None,
                   with_embeddings=False,
                   ramdom_ways_shots=False
                   ):
        print(
            "$$$$$$$$$$$$$ setDataset:\n"
            "$$$$$$$$$$$$$ data_dir_path= {}\n"
            "$$$$$$$$$$$$$ train_episode_num={}, train_batch={}, ways={}, shots={}\n"
            "$$$$$$$$$$$$$ mix_up={}, augment={}\n"
            "$$$$$$$$$$$$$ test_episode_num={}, test_train_shot={}, test_batch={}, test_shots={}".format(data_dir_path,
                                                                                                         train_episode_num,
                                                                                                         train_batch,
                                                                                                         ways, shots,
                                                                                                         mix_up,
                                                                                                         augment,
                                                                                                         test_episode_num,
                                                                                                         test_train_shot,
                                                                                                         test_batch,
                                                                                                         test_shots))

        self.data_dir_path = data_dir_path
        self.dataloader = MiniImageNetDataLoader_v2(data_dir_path=self.data_dir_path,
                                                    embedding_dir_path=embedding_dir_path,
                                                    )

        # Z_block, cluster_block = self.dataloader.get_embedding(self, redo=False, re_cluster=False, anchor_num=10)
        # self.predictor.updateAnchors(tf.reduce_mean(cluster_block, 1))

        if ramdom_ways_shots is False:
            if with_embeddings:
                self.dataloader.get_embedding(self, redo=False, re_cluster=False, anchor_num=10)
            self.meta_test_ds, meta_test_name_projector = self.dataloader.get_dataset(phase='test', way_num=ways,
                                                                                      shot_num=shots,
                                                                                      episode_test_sample_num=test_shots,
                                                                                      episode_num=test_episode_num,
                                                                                      batch=test_batch,
                                                                                      with_embeddings=with_embeddings)
        else:
            meta_test_ds = None
            task_setting_list = []
            min_ways = 5
            max_ways = 51
            for _ in range(test_episode_num):
                real_ways = random.choice(range(min_ways, max_ways))
                real_shots = max(int(min(500 // real_ways, 100) * random.uniform(0, 1.)), 1)
                # real_shots = max(int(500 // real_ways), 1)
                # real_ways = 5
                # real_shots = 5

                for __ in range(10000):
                    try:
                        meta_test_ds_, meta_test_name_projector = self.dataloader.get_dataset(phase='test',
                                                                                              way_num=real_ways,
                                                                                              shot_num=real_shots,
                                                                                              episode_test_sample_num=test_shots,
                                                                                              episode_num=1,
                                                                                              batch=1,
                                                                                              augment=False,
                                                                                              with_embeddings=False)
                        break
                    except:
                        print(_, real_ways, real_shots)
                        full_traceback_info = traceback.format_exc()
                        if -1 != full_traceback_info.find("random.sample(folders, way_num)"):
                            real_ways = real_ways - 1
                            max_ways = max_ways - 1
                        elif -1 != full_traceback_info.find("random.sample(image_list, num_samples_per_class)"):
                            real_shots = real_shots - 1
                        continue
                task_setting_list.append((_, real_ways, real_shots))
                print((_, real_ways, real_shots))
                if meta_test_ds is None:
                    meta_test_ds = meta_test_ds_
                else:
                    meta_test_ds = meta_test_ds.concatenate(meta_test_ds_)
            print(task_setting_list)

            self.meta_test_ds = meta_test_ds.prefetch(buffer_size=5)

        self.steps_per_epoch = train_episode_num // train_batch

        for it_try in range(test_train_shot):
            try:
                self.meta_train_ds, meta_train_name_projector = self.dataloader.get_dataset(phase='train',
                                                                                            way_num=ways,
                                                                                            shot_num=shots,
                                                                                            episode_test_sample_num=test_train_shot - it_try,
                                                                                            episode_num=train_episode_num,
                                                                                            batch=train_batch,
                                                                                            augment=augment,
                                                                                            mix_up=mix_up,
                                                                                            epochs=train_epoch,
                                                                                            with_embeddings=False,
                                                                                            # putback=True
                                                                                            )
                self.meta_train_ds = self.meta_train_ds.prefetch(10)
                break

            except:
                traceback.print_exc()
                continue

        print("test_train_shot : ", test_train_shot - it_try)
        if test_train_shot - it_try == 1:
            return
        image_list, label_list, global_label_depth = self.dataloader.generate_origin_data_list("train")
        self.all_train_ds = tf.data.Dataset.from_tensor_slices((image_list, label_list))

        resize_size = 92
        center_crop_size = 84
        batch = 512

        # resize_size = 224
        # center_crop_size = 224
        # batch = 64

        @tf.function
        def process(path, global_label, global_label_depth):
            image = tf.io.read_file(path)  # 根据路径读取图片
            image = tf.image.decode_jpeg(image, channels=3)  # 图片解码
            image = tf.cast(image, dtype=tf.float32) / 255.
            image = tf.image.resize(image, [resize_size, resize_size])
            image = tf.image.central_crop(image, center_crop_size / resize_size)
            image = tf.image.resize(image, [center_crop_size, center_crop_size])
            global_label = tf.one_hot(global_label, axis=-1, depth=global_label_depth)
            return image, global_label, path

        self.all_train_ds = self.all_train_ds.map(partial(process, global_label_depth=global_label_depth),
                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch)

        ###
        if not cross_domain:
            return
        print("~~~~@@@@!!!!!!~~~~~cross_domain open~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        seed = 13
        random.seed(seed)
        self.dataloader_cross = DataLoader(
            data_dir_path="/data/giraffe/0_FSL/data/CUB_200_2011/CUB_200_2011/processed_images_224_crop")
        self.meta_test_ds, meta_test_name_projector = self.dataloader_cross.get_dataset(phase='test', way_num=ways,
                                                                                        shot_num=shots,
                                                                                        episode_test_sample_num=test_shots,
                                                                                        episode_num=test_episode_num,
                                                                                        batch=test_batch,
                                                                                        augment=False)

    def train(self, num_base_embbeding, num_expresive, lr=0.001,
              anchor_num=8,
              embedding_path="/data/di/groupcs/dataset/classification/20220506_FSL_data/data/processed_images_224",
              data_name="",
              total_epoch=100, debug=False):

        scheduled_lrs = WarmUpCosine(
            learning_rate_base=lr,
            total_steps=total_epoch * self.steps_per_epoch,
            warmup_learning_rate=0.000001,
            warmup_steps=self.steps_per_epoch * 3,
        )
        # self.opt1 = tf.keras.optimizers.SGD(scheduled_lrs, momentum=0.9, nesterov=True, name="opt1")
        # self.opt1 = Adan(lr, name="opt1")
        # scheduled_lrs = WarmUpStep(
        #     learning_rate_base=lr,]
        #     warmup_learning_rate=0.00001,
        #     warmup_steps=self.steps_per_epoch * 3,
        # )
        self.opt1 = tf.optimizers.Adam(scheduled_lrs, name="opt1")
        self.train_step = self.meta_train_step
        self.compile(self.opt1)
        if debug:
            for data in self.meta_train_ds:
                self.train_step(data)
            for data in self.meta_test_ds:
                self.test_step(data)
        monitor_name = "val_{}".format("acc")
        monitor_cmp = "max"
        monitor_name = "val_mq_acc_base"
        monitor_name = "val_mq_acc_0"
        monitor_name = "val_mq_acc_{}".format(self.iterNum)
        monitor_cmp = "max"
        cur_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = self.name
        log_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_base_path = os.path.join(embedding_path, "{}_ckpts/".format(name))
        print(log_base_path, ckpt_base_path)
        filepath = os.path.join(ckpt_base_path,
                                "model_{}_best.h5".format(data_name))
        os.makedirs(ckpt_base_path, exist_ok=True)

        tensorboard_save = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_base_path, '{}_logs/{}'.format(name, cur_date)),
            profile_batch=0, )
        checkpoint_save = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                             verbose=1, monitor=monitor_name,
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode=monitor_cmp)

        callbacks = [
            checkpoint_save,
            tensorboard_save,
            PrintCallback(),
            # EmbeddingDict(num_base_embbeding=num_base_embbeding, num_expresive=num_expresive,
            #               weights=None,
            #               lr=0.01,
            #               anchor_num=anchor_num,
            #               embedding_path=embedding_path,
            #               episode_num=3000,
            #               total_epoch=50,
            #               validation_freq=1,
            #               cache=False,
            #               ),
            # tf.keras.callbacks.EarlyStopping(monitor_name, 0, 5, mode=monitor_cmp),
        ]
        # self.get_category_embedding(save_dir=embedding_path)
        # dataloaderEmb = DataLoaderEmbedding(data_dir_path=embedding_path)
        # self.embedding_achor_dict, self.embeddingAnchors, self.embeddingLabels = dataloaderEmb.generate_anchors(
        #     "train",
        #     anchor_num=anchor_num)
        # self.evaluate(self.meta_test_ds)
        # exit()
        self.fit(self.meta_train_ds.repeat(), epochs=total_epoch,
                 steps_per_epoch=self.steps_per_epoch,
                 validation_data=self.meta_test_ds,
                 callbacks=callbacks, initial_epoch=0)
        #
        if filepath is not None:
            loadStatu = model.load_weights(filepath, by_name=True, skip_mismatch=True)
            print("train stage done. loading the best model:{},\n loadStatu {}".format(filepath, loadStatu))

    def test(self):
        self.compile(tf.keras.optimizers.Adam(0.0001))
        self.evaluate(self.meta_test_ds)


if len(tf.config.list_physical_devices('GPU')) >= 1:
    multi_gpu = True
else:
    multi_gpu = False

num_base_embbeding = 1
num_expresive = 20

data_name = "FC100"
data_name = "miniImagenet"
# data_name = "meta_inat"
# data_name = "tiere meta_inat"
data_name = "tieredImagenet"
# data_name = "CUB"
# data_name = "ilsvrc_2012"

if data_name == "ilsvrc_2012":
    data_dir_path = ["/204/home/zzy/0_FSL/meta-dataset/ILSVRC_224_RECORDS/ilsvrc_2012",
                     "/204/home/zzy/0_FSL/meta-dataset/omniglot_RECORDS/omniglot",
                     "/204/home/zzy/0_FSL/meta-dataset/aircraft_RECORDS/aircraft",
                     "/204/home/zzy/0_FSL/meta-dataset/cu_birds_RECORDS/cu_birds",
                     "/204/home/zzy/0_FSL/meta-dataset/dtd_RECORDS/dtd",
                     "/204/home/zzy/0_FSL/meta-dataset/fungi_RECORDS/fungi",
                     "/204/home/zzy/0_FSL/meta-dataset/mscoco_RECORDS/mscoco",
                     "/204/home/zzy/0_FSL/meta-dataset/quickdraw_RECORDS/quickdraw",
                     "/204/home/zzy/0_FSL/meta-dataset/vgg_flower_RECORDS/vgg_flower",
                     "/204/home/zzy/0_FSL/meta-dataset/traffic_sign_RECORDS/traffic_sign"]
    data_path = data_dir_path[-9]
    if not os.path.exists(data_path):
        data_path = data_path.replace("/204/", "/")
    weigths_path = "pre-trained_model/metadataset_tiere_pretrained_model_e044-l 0.88638.h5"

    seed = 15
    random.seed(seed)
    num_class = 712
    episode_num = 2400
    anchor_num = 1
    backbone = "resnet_12"

elif data_name == "FC100":
    data_path = "/data/giraffe/0_FSL/data/FC100"
    weigths_path = "pre-trained_model/FC100_pretrained_model_e385-l 0.62184.h5"
    weigths_path = "pre-trained_model/PN_model_FC100_best.h5"

    seed = 15
    random.seed(seed)
    num_class = 60
    if not os.path.exists(data_path):
        data_path = "/data/di/groupcs/dataset/classification/20220506_FSL_data/data/FC100"
        seed = 50
        random.seed(seed)
    if not os.path.exists(data_path):
        data_path = "/data2/giraffe/0_FSL/data/FC100"
    episode_num = 12000
    anchor_num = 1
    backbone = "resnet_12"
elif data_name == "miniImagenet":
    data_path = "/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224"
    weigths_path = "pre-trained_model/mini_imagenet_pretrained_82962.h5"
    # weigths_path = None
    # weigths_path = "/home/zzy/0_FSL/SE_FS_test_ckpts/model_miniImagenet_best.h5"
    # weigths_path = "pre-trained_model/PN_model_miniImagenet_best.h5"

    seed = 44
    random.seed(seed)
    num_class = 64
    if not os.path.exists(data_path):
        data_path = "/data2/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224"

    if not os.path.exists(data_path):
        data_path = "/data/di/groupcs/dataset/classification/20220506_FSL_data/data/processed_images_224"
        seed = 50
        random.seed(seed)

    episode_num = 12000
    anchor_num = 1
    backbone = "resnet_12"


elif data_name == "tieredImagenet":
    data_path = "/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224"
    weigths_path = "pre-trained_model/tiered_model_e335-l 0.86331.h5"
    # weigths_path = "/home/zzy/0_FSL/model_tieredImagenet_best.h5"

    seed = 35
    random.seed(seed)
    num_class = 351
    if not os.path.exists(data_path):
        data_path = "/data2/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224"
    if not os.path.exists(data_path):
        data_path = "/data/di/groupcs/dataset/classification/20220506_FSL_data/data/tiered_imagenet_tools/tiered_imagenet_224"
        seed = 50
        random.seed(seed)

    episode_num = 2400
    anchor_num = 1
    backbone = "resnet_12"

elif data_name == "CUB":
    data_path = "/data/giraffe/0_FSL/data/CUB_200_2011/CUB_200_2011/processed_images_224_crop"
    weigths_path = "pre-trained_model/CUB_model_e139-l 0.90518.h5"
    # weigths_path = "pre-trained_model/SE_FS_ckpts/model_CUB_best.h5"
    # weigths_path = "pre-trained_model/PN_model_CUB_best.h5"

    seed = 13
    random.seed(seed)
    num_class = 100
    if not os.path.exists(data_path):
        data_path = "/data/di/groupcs/dataset/classification/20220506_FSL_data/data/CUB_200_2011/CUB_200_2011/processed_images_224_crop"
        seed = 50
        random.seed(seed)
    if not os.path.exists(data_path):
        data_path = "/data2/giraffe/0_FSL/data/CUB_200_2011/CUB_200_2011/processed_images_224_crop"
    episode_num = 2400
    anchor_num = 1
    backbone = "resnet_12"

elif data_name == "meta_inat":
    data_path = "/data/giraffe/0_FSL/data/inat2017/meta_iNat"
    if not os.path.exists(data_path):
        data_path = "/data2/giraffe/0_FSL/data/inat2017/meta_iNat"
    weigths_path = "pre-trained_model/meta_inat_model_e154-l 0.68938.h5"
    weigths_path = "pre-trained_model/model_meta_inat_best.h5"
    # weigths_path = "/data/giraffe/0_FSL/NEE_ckpts/model_e180-l 0.69387.h5"
    # weigths_path = "/data/giraffe/0_FSL/SE_FS_dev_embedding_anchor_num1_num_base_embbeding1_num_expresive20_meta_inat/SE_FS_dev_ckpts/model_meta_inat_best.h5"

    seed = 22
    random.seed(seed)
    num_class = 908
    episode_num = 12000
    anchor_num = 1
    backbone = "conv4"

elif data_name == "tiere meta_inat":
    data_path = "/data/giraffe/0_FSL/data/inat2017/tiered_meta_iNat"
    if not os.path.exists(data_path):
        data_path = "/data2/giraffe/0_FSL/data/inat2017/tiered_meta_iNat"
    weigths_path = "pre-trained_model/tiere_meta_inat_model_e155-l 0.61009.h5"
    # weigths_path = "pre-trained_model/model_tiere meta_inat_best.h5"
    # weigths_path = "pre-trained_model/PN_model_tiere meta_inat_best.h5"
    # weigths_path = "/data/giraffe/0_FSL/SE_FS_dev_embedding_anchor_num1_num_base_embbeding1_num_expresive20_tiere meta_inat/SE_FS_dev_ckpts/model_tiere meta_inat_best.h5"

    seed = 17
    random.seed(seed)
    num_class = 781
    episode_num = 12000
    anchor_num = 1
    backbone = "conv4"

else:
    pass

# backbone = "CLIP_ResNet"
# backbone = "CLIP_VIT"
if backbone in ["CLIP_ResNet", "CLIP_VIT"]:
    imageshape = (224, 224, 3)
else:
    imageshape = (84, 84, 3)

if multi_gpu is True:
    print("using multi_gpu!")
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = FSLModel(imageshape=imageshape, num_base_embbeding=num_base_embbeding, num_expresive=num_expresive,
                         anchor_num=anchor_num,
                         num_class=num_class, backbone=backbone)
else:
    print("using  one gpu!")
    model = FSLModel(imageshape=imageshape, num_base_embbeding=num_base_embbeding, num_expresive=num_expresive,
                     anchor_num=anchor_num,
                     num_class=num_class, backbone=backbone)

embedding_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "{}_embedding_anchor_num{}_num_base_embbeding{}_num_expresive{}_{}".format(model.name,
                                                                                                         anchor_num,
                                                                                                         num_base_embbeding,
                                                                                                         num_expresive,
                                                                                                         data_name))
print("～@@@@ current anchor_num is {}".format(anchor_num))
print("～@@@@ current num_base_embbeding is {}".format(num_base_embbeding))
print("～@@@@ current num_expresive is {}".format(num_expresive))
print("～@@@@ current Dataset is {}".format(data_name))
print("～@@@@ current data_path is {}".format(data_path))
print("～@@@@ current weigths_path is {}".format(weigths_path))
print("～@@@@ current embedding_path is {}".format(embedding_path))
print("～@@@@ current episode_num is {}".format(episode_num))
cross_domain = False
debug = False
show = False
show = True
# debug = True
# cache = False
# cache = True
episode_num = 2400
bigModel_embedding_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       "{}_{}_{}_{}_{}".format("CLIP_VIT", data_name, *list((224, 224, 3))))
print(bigModel_embedding_path)
# embedding_path = bigModel_embedding_path
# data_path = "/data2/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224"
# data_name = "tieredImagenet"
# weigths_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#                             "{}_embedding_anchor_num{}_num_base_embbeding{}_num_expresive{}_{}/fine_tune_{}_best.h5".format(
#                                 model.name,
#                                 anchor_num,
#                                 num_base_embbeding,
#                                 num_expresive,
#                                 data_name, data_name))
if debug or show:
    multi_gpu = False
    embedding_path = embedding_path.replace("/data/", "/data2/")
    bigModel_embedding_path = bigModel_embedding_path.replace("/data/", "/data2/")
    weigths_path = weigths_path.replace("/data/", "/data2/")

if weigths_path is not None:
    model.load_weights(weigths_path, by_name=True, skip_mismatch=True)

    # with mirrored_strategy.scope():
    #     model.setDataset(data_path, train_batch=6, ways=5, shots=5, test_train_shot=5, test_shots=15, mix_up=False,
    #                      test_batch=6,
    #                      test_episode_num=600, train_epoch=1, cross_domain=cross_domain)
    #     model.E_train()
if debug or show:
    if show:
        model.setDataset(data_path, train_batch=1, ways=3, shots=1, test_train_shot=6, test_shots=100, mix_up=False,
                         test_batch=1, augment=False,
                         test_episode_num=60, train_epoch=1, cross_domain=False, embedding_dir_path=embedding_path)
    else:
        model.setDataset(data_path, train_batch=2, ways=5, shots=5, test_train_shot=6, test_shots=5, mix_up=False,
                         test_batch=1, augment=False,
                         test_episode_num=60, train_epoch=1, cross_domain=False, embedding_dir_path=embedding_path)

else:
    model.setDataset(data_path, train_batch=6, ways=5, shots=5, test_train_shot=45, test_shots=15, mix_up=False,
                     test_batch=6, augment=False,
                     train_episode_num=episode_num,
                     test_episode_num=600, train_epoch=1, cross_domain=cross_domain,
                     embedding_dir_path=embedding_path)
#

if show:
    try:

        weigths_path = os.path.join(embedding_path,
                                    "fine_tune_{}_best.h5".format(data_name))
        #
        # weigths_path = "/data/giraffe/0_FSL/TRSN/pre-trained_model/protobank_fine_tune_tieredImagenet_best.h5"

        # weigths_path = "/data2/giraffe/0_FSL/SE_FS_dev_embedding_anchor_num1_num_base_embbeding1_num_expresive20_miniImagenet/SE_FS_dev_ckpts/model_miniImagenet_best.h5"
        # weigths_path = "/204/home/zzy/0_FSL/SE_FS_dev_embedding_anchor_num1_num_base_embbeding1_num_expresive20_miniImagenet/fine_tune_miniImagenet_best.h5"
        # weigths_path = "/204/home/zzy/0_FSL/SE_FS_dev_ckpts/model_miniImagenet_best.h5"

        if weigths_path is not None:
            model.load_weights(weigths_path, by_name=True, skip_mismatch=True)
        tf.print(model.predictor.T)
        tf.print(model.predictor.T2)
        model.show()
    except:
        traceback.print_exc()

# model.get_category_embedding(bigModel_embedding_path)
# model.test();
# exit()
if multi_gpu is True:
    with mirrored_strategy.scope():
        model.fine_tune(lr=0.0001,
                        num_base_embbeding=num_base_embbeding,
                        num_expresive=num_expresive,
                        anchor_num=anchor_num,
                        embedding_path=embedding_path,
                        multiView_embedding_path=[],
                        episode_num=3000,
                        total_epoch=50,
                        cache=True,
                        validation_freq=2,
                        debug=debug,
                        )
else:
    model.fine_tune(lr=0.1,
                    num_base_embbeding=num_base_embbeding,
                    num_expresive=num_expresive,
                    anchor_num=anchor_num,
                    embedding_path=embedding_path,
                    multiView_embedding_path=[],
                    episode_num=3000,
                    total_epoch=50,
                    cache=not False,
                    validation_freq=5,
                    debug=debug,
                    )
exit()
if multi_gpu is True:
    dataloaderEmb = model.loadEmbeddingAnchors(anchor_num, embedding_path, not False)
    for _ in range(1):
        with mirrored_strategy.scope():
            # model.fine_tune(lr=0.0001,
            #                 num_base_embbeding=num_base_embbeding,
            #                 num_expresive=num_expresive,
            #                 anchor_num=anchor_num,
            #                 embedding_path=embedding_path,
            #                 episode_num=3000,
            #                 total_epoch=3,
            #                 cache=True,
            #                 validation_freq=1,
            #                 debug=debug,
            #                 )

            best_model_filepath = model.train(lr=0.00001,
                                              num_base_embbeding=num_base_embbeding,
                                              num_expresive=num_expresive,
                                              anchor_num=anchor_num,
                                              embedding_path=embedding_path,
                                              total_epoch=50,
                                              data_name=data_name,
                                              debug=debug,
                                              )


else:

    best_model_filepath = model.train(lr=0.00005,
                                      num_base_embbeding=num_base_embbeding,
                                      num_expresive=num_expresive,
                                      anchor_num=anchor_num,
                                      embedding_path=embedding_path,
                                      total_epoch=50,
                                      data_name=data_name,
                                      debug=debug,
                                      )

'''
minimagenet
 CLIP res50     mq_acc_base: 0.9272 - mq_acc_base_1s: 0.7860 - query_loss: 0.2661 - intra_loss: 0.2072 - inter_loss: 0.4412 - entropy: 0.4724 - InfoNCE: 0.2661 
 CLIP VIT-L/14: mq_acc_base: 0.9777 - mq_acc_base_1s: 0.8345 - query_loss: 0.2258 - intra_loss: 0.1769 - inter_loss: 0.3575 - entropy: 0.5794 - InfoNCE: 0.2258
cub
 CLIP res50     mq_acc_base: 0.7193 - mq_acc_base_1s: 0.5077 - query_loss: 1.0303 - intra_loss: 0.1187 - inter_loss: 0.1715 - entropy: 1.3721 - InfoNCE: 1.0303 
 CLIP VIT-L/14: mq_acc_base: 0.9587 - mq_acc_base_1s: 0.8241 - query_loss: 0.5218 - intra_loss: 0.0880 - inter_loss: 0.1926 - entropy: 1.0805 - InfoNCE: 0.5218
tier
 CLIP res50     mq_acc_base: 0.8980 - mq_acc_base_1s: 0.7620 - query_loss: 0.3258 - intra_loss: 0.1918 - inter_loss: 0.4077 - entropy: 0.5297 - InfoNCE: 0.3258
 CLIP VIT-L/14: mq_acc_base: 0.9663 - mq_acc_base_1s: 0.8200 - query_loss: 0.2671 - intra_loss: 0.1609 - inter_loss: 0.3317 - entropy: 0.6408 - InfoNCE: 0.2671
FC100
 CLIP res50     mq_acc_base: 0.5057 - mq_acc_base_1s: 0.3500 - query_loss: 1.4026 - intra_loss: 0.0578 - inter_loss: 0.0738 - entropy: 1.5624 - InfoNCE: 1.4026 
 CLIP VIT-L/14: mq_acc_base: 0.6597 - mq_acc_base_1s: 0.4372 - query_loss: 1.1439 - intra_loss: 0.1156 - inter_loss: 0.1549 - entropy: 1.4560 - InfoNCE: 1.1439

FC100 bgr to rgb
 CLIP VIT-L/14: mq_acc_base: 0.7358 - mq_acc_base_1s: 0.5169 - query_loss: 0.9887 - intra_loss: 0.1193 - inter_loss: 0.1744 - entropy: 1.3733 - InfoNCE: 0.9887   
meta_inat
 CLIP VIT-L/14: mq_acc_base: 0.9288 - mq_acc_base_1s: 0.7777 - query_loss: 0.3784 - intra_loss: 0.1419 - inter_loss: 0.2926 - entropy: 0.7677 - InfoNCE: 0.3784
tiere meta_inat
 CLIP VIT-L/14: mq_acc_base: 0.8518 - mq_acc_base_1s: 0.6576 - query_loss: 0.6615 - intra_loss: 0.1183 - inter_loss: 0.2211 - entropy: 1.0764 - InfoNCE: 0.6615
'''
'''
meta_inat       - mq_acc_0: 0.8059 - mq_acc_1: 0.8204 - mq_acc_2: 0.8245 - mq_acc_3: 0.8263 - mq_acc_4: 0.8269 - mq_acc_5: 0.8282 - mq_acc_6: 0.8284 - mq_acc_7: 0.8283 - mq_acc_8: 0.8286 - mq_acc_9: 0.8283 - mq_acc_10: 0.8281 - mq_acc_11: 0.8280 - mq_acc_12: 0.8279 - mq_acc_13: 0.8278 - mq_acc_14: 0.8273 - mq_acc_15: 0.8270 - mq_acc_16: 0.8267 - mq_acc_17: 0.8261 - mq_acc_18: 0.8263 - mq_acc_19: 0.8263 - mq_acc_20: 0.8264 - mq_acc_1s_0: 0.6300 - mq_acc_1s_1: 0.6622 - mq_acc_1s_2: 0.6734 - mq_acc_1s_3: 0.6815 - mq_acc_1s_4: 0.6859 - mq_acc_1s_5: 0.6883 - mq_acc_1s_6: 0.6909 - mq_acc_1s_7: 0.6935 - mq_acc_1s_8: 0.6945 - mq_acc_1s_9: 0.6952 - mq_acc_1s_10: 0.6958 - mq_acc_1s_11: 0.6966 - mq_acc_1s_12: 0.6969 - mq_acc_1s_13: 0.6976 - mq_acc_1s_14: 0.6977 - mq_acc_1s_15: 0.6980 - mq_acc_1s_16: 0.6985 - mq_acc_1s_17: 0.6987 - mq_acc_1s_18: 0.6989 - mq_acc_1s_19: 0.6989 - mq_acc_1s_20: 0.6990
tiere meta_inat - mq_acc_0: 0.6691 - mq_acc_1: 0.6719 - va_mq_acc_2: 0.6722 - mq_acc_3: 0.6729 - mq_acc_4: 0.6731 - mq_acc_5: 0.6731 - mq_acc_6: 0.6731 - mq_acc_7: 0.6732 - mq_acc_8: 0.6730 - mq_acc_9: 0.6724 - mq_acc_10: 0.6719 - mq_acc_11: 0.6715 - mq_acc_12: 0.6710 - mq_acc_13: 0.6702 - mq_acc_14: 0.6701 - mq_acc_15: 0.6697 - mq_acc_16: 0.6695 - mq_acc_17: 0.6691 - mq_acc_18: 0.6686 - mq_acc_19: 0.6684 - mq_acc_20: 0.6684 - mq_acc_1s_0: 0.4748 - mq_acc_1s_1: 0.4764 - mq_acc_1s_2: 0.4787 - mq_acc_1s_3: 0.4809 - mq_acc_1s_4: 0.4819 - mq_acc_1s_5: 0.4840 - mq_acc_1s_6: 0.4841 - mq_acc_1s_7: 0.4848 - mq_acc_1s_8: 0.4854 - mq_acc_1s_9: 0.4854 - mq_acc_1s_10: 0.4856 - mq_acc_1s_11: 0.4861 - mq_acc_1s_12: 0.4853 - mq_acc_1s_13: 0.4851 - mq_acc_1s_14: 0.4849 - mq_acc_1s_15: 0.4845 - mq_acc_1s_16: 0.4846 - mq_acc_1s_17: 0.4844 - mq_acc_1s_18: 0.4837 - mq_acc_1s_19: 0.4835 - mq_acc_1s_20: 0.4832
imagenet        - mq_acc_0: 0.8575 - mq_acc_1: 0.8587 - mq_acc_2: 0.8587 - mq_acc_3: 0.8587 - mq_acc_4: 0.8587 - mq_acc_5: 0.8592 - mq_acc_6: 0.8594 - mq_acc_7: 0.8599 - mq_acc_8: 0.8602 - mq_acc_9: 0.8604 - mq_acc_10: 0.8607 - mq_acc_11: 0.8609 - mq_acc_12: 0.8607 - mq_acc_13: 0.8606 - mq_acc_14: 0.8608 - mq_acc_15: 0.8606 - mq_acc_16: 0.8604 - mq_acc_17: 0.8602 - mq_acc_18: 0.8600 - mq_acc_19: 0.8596 - mq_acc_20: 0.8592 - mq_acc_1s_0: 0.6925 - mq_acc_1s_1: 0.6925 - mq_acc_1s_2: 0.6926 - mq_acc_1s_3: 0.6931 - mq_acc_1s_4: 0.6947 - mq_acc_1s_5: 0.6964 - mq_acc_1s_6: 0.6977 - mq_acc_1s_7: 0.6986 - mq_acc_1s_8: 0.6998 - mq_acc_1s_9: 0.7009 - mq_acc_1s_10: 0.7019 - mq_acc_1s_11: 0.7027 - mq_acc_1s_12: 0.7032 - mq_acc_1s_13: 0.7037 - mq_acc_1s_14: 0.7038 - mq_acc_1s_15: 0.7042 - mq_acc_1s_16: 0.7047 - mq_acc_1s_17: 0.7044 - mq_acc_1s_18: 0.7047 - mq_acc_1s_19: 0.7043 - mq_acc_1s_20: 0.7042  
tieredImagenet  - mq_acc_0: 0.8699 - mq_acc_1: 0.8737 - mq_acc_2: 0.8737 - mq_acc_3: 0.8739 - mq_acc_4: 0.8743 - mq_acc_5: 0.8751 - mq_acc_6: 0.8763 - mq_acc_7: 0.8773 - mq_acc_8: 0.8785 - mq_acc_9: 0.8800 - mq_acc_10: 0.8811 - mq_acc_11: 0.8822 - mq_acc_12: 0.8829 - mq_acc_13: 0.8833 - mq_acc_14: 0.8839 - mq_acc_15: 0.8845 - mq_acc_16: 0.8846 - mq_acc_17: 0.8850 - mq_acc_18: 0.8852 - mq_acc_19: 0.8854 - mq_acc_20: 0.8857 - mq_acc_1s_0: 0.6869 - mq_acc_1s_1: 0.6870 - mq_acc_1s_2: 0.6876 - mq_acc_1s_3: 0.6900 - mq_acc_1s_4: 0.6944 - mq_acc_1s_5: 0.7008 - mq_acc_1s_6: 0.7075 - mq_acc_1s_7: 0.7139 - mq_acc_1s_8: 0.7194 - mq_acc_1s_9: 0.7237 - mq_acc_1s_10: 0.7271 - mq_acc_1s_11: 0.7302 - mq_acc_1s_12: 0.7322 - mq_acc_1s_13: 0.7349 - mq_acc_1s_14: 0.7357 - mq_acc_1s_15: 0.7365 - mq_acc_1s_16: 0.7374 - mq_acc_1s_17: 0.7378 - mq_acc_1s_18: 0.7377 - mq_acc_1s_19: 0.7377 - mq_acc_1s_20: 0.7376
FC100           - mq_acc_0: 0.6445 - mq_acc_1: 0.6465 - mq_acc_2: 0.6466 - mq_acc_3: 0.6470 - mq_acc_4: 0.6473 - mq_acc_5: 0.6475 - mq_acc_6: 0.6470 - mq_acc_7: 0.6471 - mq_acc_8: 0.6472 - mq_acc_9: 0.6472 - mq_acc_10: 0.6470 - mq_acc_11: 0.6466 - mq_acc_12: 0.6464 - mq_acc_13: 0.6462 - mq_acc_14: 0.6458 - mq_acc_15: 0.6456 - mq_acc_16: 0.6455 - mq_acc_17: 0.6454 - mq_acc_18: 0.6453 - mq_acc_19: 0.6452 - mq_acc_20: 0.6451 - mq_acc_1s_0: 0.4616 - mq_acc_1s_1: 0.4618 - mq_acc_1s_2: 0.4622 - mq_acc_1s_3: 0.4640 - mq_acc_1s_4: 0.4654 - mq_acc_1s_5: 0.4664 - mq_acc_1s_6: 0.4667 - mq_acc_1s_7: 0.4671 - mq_acc_1s_8: 0.4678 - mq_acc_1s_9: 0.4682 - mq_acc_1s_10: 0.4688 - mq_acc_1s_11: 0.4689 - mq_acc_1s_12: 0.4692 - va$_mq_acc_1s_13: 0.4694 - mq_acc_1s_14: 0.4695 - mq_acc_1s_15: 0.4694 - mq_acc_1s_16: 0.4696 - mq_acc_1s_17: 0.4696 - mq_acc_1s_18: 0.4698 - mq_acc_1s_19: 0.4700 - mq_acc_1s_20: 0.4700 


Omglot          - mq_acc_base: 0.9042 - mq_acc_base_1s: 0.7510 - mq_acc_0: 0.9100 - mq_acc_1: 0.9101 - mq_acc_2: 0.9102 - mq_acc_3: 0.9095 - mq_acc_4: 0.9089 - mq_acc_5: 0.9080 - mq_acc_6: 0.9070 - mq_acc_7: 0.9064 - mq_acc_8: 0.9057 - mq_acc_9: 0.9050 - mq_acc_10: 0.9048 - mq_acc_11: 0.9046 - mq_acc_12: 0.9045 - mq_acc_13: 0.9047 - mq_acc_14: 0.9045 - mq_acc_15: 0.9045 - mq_acc_16: 0.9044 - mq_acc_17: 0.9044 - mq_acc_18: 0.9044 - mq_acc_19: 0.9042 - mq_acc_20: 0.9042 - mq_acc_1s_0: 0.7538 - mq_acc_1s_1: 0.7539 - mq_acc_1s_2: 0.7543 - mq_acc_1s_3: 0.7538 - mq_acc_1s_4: 0.7537 - mq_acc_1s_5: 0.7527 - mq_acc_1s_6: 0.7516 - mq_acc_1s_7: 0.7508 - mq_acc_1s_8: 0.7501 - mq_acc_1s_9: 0.7499 - mq_acc_1s_10: 0.7497 - mq_acc_1s_11: 0.7496 - mq_acc_1s_12: 0.7497 - mq_acc_1s_13: 0.7499 - mq_acc_1s_14: 0.7502 - mq_acc_1s_15: 0.7504 - mq_acc_1s_16: 0.7507 - mq_acc_1s_17: 0.7507 - mq_acc_1s_18: 0.7507 - mq_acc_1s_19: 0.7509 - mq_acc_1s_20: 0.7510
Acraft          - mq_acc_base: 0.4158 - mq_acc_base_1s: 0.3027 - mq_acc_0: 0.4154 - mq_acc_1: 0.4152 - mq_acc_2: 0.4152 - mq_acc_3: 0.4152 - mq_acc_4: 0.4152 - mq_acc_5: 0.4153 - mq_acc_6: 0.4164 - mq_acc_7: 0.4166 - mq_acc_8: 0.4170 - mq_acc_9: 0.4173 - mq_acc_10: 0.4165 - mq_acc_11: 0.4164 - mq_acc_12: 0.4164 - mq_acc_13: 0.4163 - mq_acc_14: 0.4161 - mq_acc_15: 0.4160 - mq_acc_16: 0.4159 - mq_acc_17: 0.4158 - mq_acc_18: 0.4156 - mq_acc_19: 0.4157 - mq_acc_20: 0.4158 - mq_acc_1s_0: 0.3008 - mq_acc_1s_1: 0.3008 - mq_acc_1s_2: 0.3008 - mq_acc_1s_3: 0.3008 - mq_acc_1s_4: 0.3014 - mq_acc_1s_5: 0.3017 - mq_acc_1s_6: 0.3026 - mq_acc_1s_7: 0.3035 - mq_acc_1s_8: 0.3039 - mq_acc_1s_9: 0.3038 - mq_acc_1s_10: 0.3041 - mq_acc_1s_11: 0.3038 - mq_acc_1s_12: 0.3032 - mq_acc_1s_13: 0.3035 - mq_acc_1s_14: 0.3035 - mq_acc_1s_15: 0.3030 - mq_acc_1s_16: 0.3029 - mq_acc_1s_17: 0.3031 - mq_acc_1s_18: 0.3031 - mq_acc_1s_19: 0.3029 - mq_acc_1s_20: 0.3027
CUB             - mq_acc_base: 0.8203 - mq_acc_base_1s: 0.6026 - mq_acc_0: 0.8167 - mq_acc_1: 0.8178 - mq_acc_2: 0.8178 - mq_acc_3: 0.8179 - mq_acc_4: 0.8188 - mq_acc_5: 0.8198 - mq_acc_6: 0.8204 - mq_acc_7: 0.8205 - mq_acc_8: 0.8210 - mq_acc_9: 0.8212 - mq_acc_10: 0.8215 - mq_acc_11: 0.8214 - mq_acc_12: 0.8214 - mq_acc_13: 0.8214 - mq_acc_14: 0.8214 - mq_acc_15: 0.8215 - mq_acc_16: 0.8212 - mq_acc_17: 0.8209 - mq_acc_18: 0.8205 - mq_acc_19: 0.8205 - mq_acc_20: 0.8203 - mq_acc_1s_0: 0.5825 - mq_acc_1s_1: 0.5825 - mq_acc_1s_2: 0.5826 - mq_acc_1s_3: 0.5835 - mq_acc_1s_4: 0.5864 - mq_acc_1s_5: 0.5890 - mq_acc_1s_6: 0.5916 - mq_acc_1s_7: 0.5937 - mq_acc_1s_8: 0.5950 - mq_acc_1s_9: 0.5969 - mq_acc_1s_10: 0.5988 - mq_acc_1s_11: 0.6001 - mq_acc_1s_12: 0.6014 - mq_acc_1s_13: 0.6016 - mq_acc_1s_14: 0.6019 - mq_acc_1s_15: 0.6024 - mq_acc_1s_16: 0.6025 - mq_acc_1s_17: 0.6024 - mq_acc_1s_18: 0.6027 - mq_acc_1s_19: 0.6027 - mq_acc_1s_20: 0.6026
DTD             - mq_acc_base: 0.6882 - mq_acc_base_1s: 0.5078 - mq_acc_0: 0.6758 - mq_acc_1: 0.6819 - mq_acc_2: 0.6844 - mq_acc_3: 0.6868 - mq_acc_4: 0.6880 - mq_acc_5: 0.6892 - mq_acc_6: 0.6898 - mq_acc_7: 0.6905 - mq_acc_8: 0.6908 - mq_acc_9: 0.6906 - mq_acc_10: 0.6907 - mq_acc_11: 0.6905 - mq_acc_12: 0.6906 - mq_acc_13: 0.6900 - mq_acc_14: 0.6899 - mq_acc_15: 0.6895 - mq_acc_16: 0.6892 - mq_acc_17: 0.6888 - mq_acc_18: 0.6887 - mq_acc_19: 0.6886 - mq_acc_20: 0.6882 - mq_acc_1s_0: 0.4947 - mq_acc_1s_1: 0.4964 - mq_acc_1s_2: 0.4996 - mq_acc_1s_3: 0.5023 - mq_acc_1s_4: 0.5041 - mq_acc_1s_5: 0.5064 - mq_acc_1s_6: 0.5076 - mq_acc_1s_7: 0.5089 - mq_acc_1s_8: 0.5089 - mq_acc_1s_9: 0.5091 - mq_acc_1s_10: 0.5091 - mq_acc_1s_11: 0.5091 - mq_acc_1s_12: 0.5087 - mq_acc_1s_13: 0.5084 - mq_acc_1s_14: 0.5087 - mq_acc_1s_15: 0.5084 - mq_acc_1s_16: 0.5079 - mq_acc_1s_17: 0.5079 - mq_acc_1s_18: 0.5077 - mq_acc_1s_19: 0.5078 - mq_acc_1s_20: 0.5078
Fungi           - mq_acc_base: 0.7145 - mq_acc_base_1s: 0.5059 - mq_acc_0: 0.7088 - mq_acc_1: 0.7107 - mq_acc_2: 0.7108 - mq_acc_3: 0.7117 - mq_acc_4: 0.7130 - mq_acc_5: 0.7140 - mq_acc_6: 0.7143 - mq_acc_7: 0.7147 - mq_acc_8: 0.7151 - mq_acc_9: 0.7152 - mq_acc_10: 0.7149 - mq_acc_11: 0.7154 - mq_acc_12: 0.7151 - mq_acc_13: 0.7150 - mq_acc_14: 0.7148 - mq_acc_15: 0.7143 - mq_acc_16: 0.7144 - mq_acc_17: 0.7143 - mq_acc_18: 0.7139 - mq_acc_19: 0.7140 - mq_acc_20: 0.7145 - mq_acc_1s_0: 0.4895 - mq_acc_1s_1: 0.4895 - mq_acc_1s_2: 0.4901 - mq_acc_1s_3: 0.4921 - mq_acc_1s_4: 0.4945 - mq_acc_1s_5: 0.4972 - mq_acc_1s_6: 0.4999 - mq_acc_1s_7: 0.5012 - mq_acc_1s_8: 0.5030 - mq_acc_1s_9: 0.5037 - mq_acc_1s_10: 0.5046 - mq_acc_1s_11: 0.5052 - mq_acc_1s_12: 0.5054 - mq_acc_1s_13: 0.5054 - mq_acc_1s_14: 0.5051 - mq_acc_1s_15: 0.5055 - mq_acc_1s_16: 0.5052 - mq_acc_1s_17: 0.5051 - mq_acc_1s_18: 0.5058 - mq_acc_1s_19: 0.5055 - mq_acc_1s_20: 0.5059
COCO            - mq_acc_base: 0.6521 - mq_acc_base_1s: 0.4759 - mq_acc_0: 0.6218 - mq_acc_1: 0.6271 - mq_acc_2: 0.6285 - mq_acc_3: 0.6320 - mq_acc_4: 0.6350 - mq_acc_5: 0.6373 - mq_acc_6: 0.6406 - mq_acc_7: 0.6426 - mq_acc_8: 0.6446 - mq_acc_9: 0.6465 - mq_acc_10: 0.6482 - mq_acc_11: 0.6490 - mq_acc_12: 0.6489 - mq_acc_13: 0.6493 - mq_acc_14: 0.6496 - mq_acc_15: 0.6500 - mq_acc_16: 0.6508 - mq_acc_17: 0.6512 - mq_acc_18: 0.6514 - mq_acc_19: 0.6517 - mq_acc_20: 0.6521 - mq_acc_1s_0: 0.4447 - mq_acc_1s_1: 0.4451 - mq_acc_1s_2: 0.4466 - mq_acc_1s_3: 0.4492 - mq_acc_1s_4: 0.4528 - mq_acc_1s_5: 0.4561 - mq_acc_1s_6: 0.4594 - mq_acc_1s_7: 0.4631 - mq_acc_1s_8: 0.4654 - mq_acc_1s_9: 0.4672 - mq_acc_1s_10: 0.4692 - mq_acc_1s_11: 0.4703 - mq_acc_1s_12: 0.4712 - mq_acc_1s_13: 0.4720 - mq_acc_1s_14: 0.4729 - mq_acc_1s_15: 0.4731 - mq_acc_1s_16: 0.4736 - mq_acc_1s_17: 0.4739 - mq_acc_1s_18: 0.4747 - mq_acc_1s_19: 0.4752 - mq_acc_1s_20: 0.4759
QDraw           - mq_acc_base: 0.7360 - mq_acc_base_1s: 0.5603 - mq_acc_0: 0.7466 - mq_acc_1: 0.7468 - mq_acc_2: 0.7462 - mq_acc_3: 0.7456 - mq_acc_4: 0.7440 - mq_acc_5: 0.7426 - mq_acc_6: 0.7418 - mq_acc_7: 0.7413 - mq_acc_8: 0.7407 - mq_acc_9: 0.7401 - mq_acc_10: 0.7402 - mq_acc_11: 0.7398 - mq_acc_12: 0.7392 - mq_acc_13: 0.7385 - mq_acc_14: 0.7380 - mq_acc_15: 0.7373 - mq_acc_16: 0.7368 - mq_acc_17: 0.7365 - mq_acc_18: 0.7364 - mq_acc_19: 0.7362 - mq_acc_20: 0.7360 - mq_acc_1s_0: 0.5584 - mq_acc_1s_1: 0.5591 - mq_acc_1s_2: 0.5601 - mq_acc_1s_3: 0.5596 - mq_acc_1s_4: 0.5601 - mq_acc_1s_5: 0.5603 - mq_acc_1s_6: 0.5598 - mq_acc_1s_7: 0.5595 - mq_acc_1s_8: 0.5595 - mq_acc_1s_9: 0.5594 - mq_acc_1s_10: 0.5597 - mq_acc_1s_11: 0.5598 - mq_acc_1s_12: 0.5600 - mq_acc_1s_13: 0.5603 - mq_acc_1s_14: 0.5604 - mq_acc_1s_15: 0.5605 - mq_acc_1s_16: 0.5605 - mq_acc_1s_17: 0.5604 - mq_acc_1s_18: 0.5603 - mq_acc_1s_19: 0.5602 - mq_acc_1s_20: 0.5603
Flower          - mq_acc_base: 0.9369 - mq_acc_base_1s: 0.7821 - mq_acc_0: 0.9360 - mq_acc_1: 0.9366 - mq_acc_2: 0.9368 - mq_acc_3: 0.9371 - mq_acc_4: 0.9376 - mq_acc_5: 0.9378 - mq_acc_6: 0.9382 - mq_acc_7: 0.9385 - mq_acc_8: 0.9385 - mq_acc_9: 0.9382 - mq_acc_10: 0.9380 - mq_acc_11: 0.9380 - mq_acc_12: 0.9382 - mq_acc_13: 0.9378 - mq_acc_14: 0.9376 - mq_acc_15: 0.9374 - mq_acc_16: 0.9374 - mq_acc_17: 0.9374 - mq_acc_18: 0.9373 - mq_acc_19: 0.9371 - mq_acc_20: 0.9369 - mq_acc_1s_0: 0.7759 - mq_acc_1s_1: 0.7759 - mq_acc_1s_2: 0.7764 - mq_acc_1s_3: 0.7791 - mq_acc_1s_4: 0.7818 - mq_acc_1s_5: 0.7839 - mq_acc_1s_6: 0.7856 - mq_acc_1s_7: 0.7856 - mq_acc_1s_8: 0.7862 - mq_acc_1s_9: 0.7863 - mq_acc_1s_10: 0.7861 - mq_acc_1s_11: 0.7859 - mq_acc_1s_12: 0.7854 - mq_acc_1s_13: 0.7848 - mq_acc_1s_14: 0.7845 - mq_acc_1s_15: 0.7838 - mq_acc_1s_16: 0.7835 - mq_acc_1s_17: 0.7828 - mq_acc_1s_18: 0.7826 - mq_acc_1s_19: 0.7822 - mq_acc_1s_20: 0.7821
Sign            - mq_acc_base: 0.7707 - mq_acc_base_1s: 0.6071 - mq_acc_0: 0.7693 - mq_acc_1: 0.7705 - mq_acc_2: 0.7708 - mq_acc_3: 0.7717 - mq_acc_4: 0.7728 - mq_acc_5: 0.7729 - mq_acc_6: 0.7727 - mq_acc_7: 0.7728 - mq_acc_8: 0.7729 - mq_acc_9: 0.7724 - mq_acc_10: 0.7724 - mq_acc_11: 0.7722 - mq_acc_12: 0.7723 - mq_acc_13: 0.7720 - mq_acc_14: 0.7719 - mq_acc_15: 0.7717 - mq_acc_16: 0.7717 - mq_acc_17: 0.7714 - mq_acc_18: 0.7710 - mq_acc_19: 0.7709 - mq_acc_20: 0.7707 - mq_acc_1s_0: 0.6036 - mq_acc_1s_1: 0.6042 - mq_acc_1s_2: 0.6054 - mq_acc_1s_3: 0.6062 - mq_acc_1s_4: 0.6072 - mq_acc_1s_5: 0.6082 - mq_acc_1s_6: 0.6084 - mq_acc_1s_7: 0.6085 - mq_acc_1s_8: 0.6083 - mq_acc_1s_9: 0.6082 - mq_acc_1s_10: 0.6080 - mq_acc_1s_11: 0.6076 - mq_acc_1s_12: 0.6073 - mq_acc_1s_13: 0.6070 - mq_acc_1s_14: 0.6071 - mq_acc_1s_15: 0.6070 - mq_acc_1s_16: 0.6071 - mq_acc_1s_17: 0.6073 - mq_acc_1s_18: 0.6070 - mq_acc_1s_19: 0.6071 - mq_acc_1s_20: 0.6071



shot 
1       - mq_acc_0: 0.6869 - mq_acc_1: 0.6870 - mq_acc_2: 0.6876 - mq_acc_3: 0.6900 - mq_acc_4: 0.6944 - mq_acc_5: 0.7008 - mq_acc_6: 0.7075 - mq_acc_7: 0.7139 - mq_acc_8: 0.7194 - mq_acc_9: 0.7237 - mq_acc_10: 0.7271 - mq_acc_11: 0.7302 - mq_acc_12: 0.7322 - mq_acc_13: 0.7349 - mq_acc_14: 0.7357 - mq_acc_15: 0.7365 - mq_acc_16: 0.7374 - mq_acc_17: 0.7378 - mq_acc_18: 0.7377 - mq_acc_19: 0.7377 - mq_acc_20: 0.7376
3       - mq_acc_0: 0.8218 - mq_acc_1: 0.8252 - mq_acc_2: 0.8258 - mq_acc_3: 0.8270 - mq_acc_4: 0.8285 - mq_acc_5: 0.8297 - mq_acc_6: 0.8320 - mq_acc_7: 0.8335 - mq_acc_8: 0.8353 - mq_acc_9: 0.8366 - mq_acc_10: 0.8382 - mq_acc_11: 0.8393 - mq_acc_12: 0.8397 - mq_acc_13: 0.8401 - mq_acc_14: 0.8407 - mq_acc_15: 0.8413 - mq_acc_16: 0.8419 - mq_acc_17: 0.8421 - mq_acc_18: 0.8426 - mq_acc_19: 0.8428 - mq_acc_20: 0.8428
5       - mq_acc_0: 0.8699 - mq_acc_1: 0.8737 - mq_acc_2: 0.8737 - mq_acc_3: 0.8739 - mq_acc_4: 0.8743 - mq_acc_5: 0.8751 - mq_acc_6: 0.8763 - mq_acc_7: 0.8773 - mq_acc_8: 0.8785 - mq_acc_9: 0.8800 - mq_acc_10: 0.8811 - mq_acc_11: 0.8822 - mq_acc_12: 0.8829 - mq_acc_13: 0.8833 - mq_acc_14: 0.8839 - mq_acc_15: 0.8845 - mq_acc_16: 0.8846 - mq_acc_17: 0.8850 - mq_acc_18: 0.8852 - mq_acc_19: 0.8854 - mq_acc_20: 0.8857 
10      - mq_acc_0: 0.8939 - mq_acc_1: 0.8957 - mq_acc_2: 0.8959 - mq_acc_3: 0.8962 - mq_acc_4: 0.8970 - mq_acc_5: 0.8979 - mq_acc_6: 0.8988 - mq_acc_7: 0.8992 - mq_acc_8: 0.9000 - mq_acc_9: 0.9006 - mq_acc_10: 0.9008 - mq_acc_11: 0.9015 - mq_acc_12: 0.9019 - mq_acc_13: 0.9025 - mq_acc_14: 0.9023 - mq_acc_15: 0.9024 - mq_acc_16: 0.9024 - mq_acc_17: 0.9025 - mq_acc_18: 0.9027 - mq_acc_19: 0.9030 - mq_acc_20: 0.9032
15      - mq_acc_0: 0.9043 - mq_acc_1: 0.9057 - mq_acc_2: 0.9058 - mq_acc_3: 0.9059 - mq_acc_4: 0.9064 - mq_acc_5: 0.9070 - mq_acc_6: 0.9078 - mq_acc_7: 0.9084 - mq_acc_8: 0.9091 - mq_acc_9: 0.9096 - mq_acc_10: 0.9101 - mq_acc_11: 0.9103 - mq_acc_12: 0.9107 - mq_acc_13: 0.9110 - mq_acc_14: 0.9112 - mq_acc_15: 0.9111 - mq_acc_16: 0.9111 - mq_acc_17: 0.9112 - mq_acc_18: 0.9115 - mq_acc_19: 0.9118 - mq_acc_20: 0.9117
20      - mq_acc_0: 0.9087 - mq_acc_1: 0.9103 - mq_acc_2: 0.9103 - mq_acc_3: 0.9106 - mq_acc_4: 0.9109 - mq_acc_5: 0.9114 - mq_acc_6: 0.9115 - mq_acc_7: 0.9116 - mq_acc_8: 0.9124 - mq_acc_9: 0.9129 - mq_acc_10: 0.9130 - mq_acc_11: 0.9136 - mq_acc_12: 0.9140 - mq_acc_13: 0.9143 - mq_acc_14: 0.9145 - mq_acc_15: 0.9149 - mq_acc_16: 0.9149 - mq_acc_17: 0.9150 - mq_acc_18: 0.9153 - mq_acc_19: 0.9153 - mq_acc_20: 0.9154 

tiere-Imagenet
beta 
0.3     mq_acc_0: 0.8699 - mq_acc_1: 0.8700 - mq_acc_2: 0.8700 - mq_acc_3: 0.8700 - mq_acc_4: 0.8700 - mq_acc_5: 0.8700 - mq_acc_6: 0.8700 - mq_acc_7: 0.8700 - mq_acc_8: 0.8700 - mq_acc_9: 0.8700 - mq_acc_10: 0.8700 - mq_acc_11: 0.8700 - mq_acc_12: 0.8700 - mq_acc_13: 0.8700 - mq_acc_14: 0.8700 - mq_acc_15: 0.8700 - mq_acc_16: 0.8700 - mq_acc_17: 0.8700 - mq_acc_18: 0.8700 - mq_acc_19: 0.8700 - mq_acc_20: 0.8700 - mq_acc_1s_0: 0.6869 - mq_acc_1s_1: 0.6869 - mq_acc_1s_2: 0.6869 - mq_acc_1s_3: 0.6869 - mq_acc_1s_4: 0.6869 - mq_acc_1s_5: 0.6869 - mq_acc_1s_6: 0.6869 - mq_acc_1s_7: 0.6869 - mq_acc_1s_8: 0.6869 - mq_acc_1s_9: 0.6869 - mq_acc_1s_10: 0.6869 - mq_acc_1s_11: 0.6869 - mq_acc_1s_12: 0.6869 - mq_acc_1s_13: 0.6869 - mq_acc_1s_14: 0.6869 - mq_acc_1s_15: 0.6869 - mq_acc_1s_16: 0.6869 - mq_acc_1s_17: 0.6869 - mq_acc_1s_18: 0.6870 - mq_acc_1s_19: 0.6870 - mq_acc_1s_20: 0.6870
0.2     mq_acc_0: 0.8699 - mq_acc_1: 0.8716 - mq_acc_2: 0.8716 - mq_acc_3: 0.8716 - mq_acc_4: 0.8716 - mq_acc_5: 0.8716 - mq_acc_6: 0.8716 - mq_acc_7: 0.8716 - mq_acc_8: 0.8716 - mq_acc_9: 0.8716 - mq_acc_10: 0.8716 - mq_acc_11: 0.8716 - mq_acc_12: 0.8716 - mq_acc_13: 0.8716 - mq_acc_14: 0.8716 - mq_acc_15: 0.8716 - mq_acc_16: 0.8716 - mq_acc_17: 0.8716 - mq_acc_18: 0.8716 - mq_acc_19: 0.8716 - mq_acc_20: 0.8716 - mq_acc_1s_0: 0.6869 - mq_acc_1s_1: 0.6869 - mq_acc_1s_2: 0.6869 - mq_acc_1s_3: 0.6869 - mq_acc_1s_4: 0.6870 - mq_acc_1s_5: 0.6870 - mq_acc_1s_6: 0.6871 - mq_acc_1s_7: 0.6873 - mq_acc_1s_8: 0.6873 - mq_acc_1s_9: 0.6874 - mq_acc_1s_10: 0.6875 - mq_acc_1s_11: 0.6876 - mq_acc_1s_12: 0.6877 - mq_acc_1s_13: 0.6877 - mq_acc_1s_14: 0.6877 - mq_acc_1s_15: 0.6878 - mq_acc_1s_16: 0.6878 - mq_acc_1s_17: 0.6879 - mq_acc_1s_18: 0.6879 - mq_acc_1s_19: 0.6879 - mq_acc_1s_20: 0.6879
0.1     mq_acc_0: 0.8699 - mq_acc_1: 0.8735 - mq_acc_2: 0.8735 - mq_acc_3: 0.8735 - mq_acc_4: 0.8735 - mq_acc_5: 0.8736 - mq_acc_6: 0.8738 - mq_acc_7: 0.8738 - mq_acc_8: 0.8741 - mq_acc_9: 0.8744 - mq_acc_10: 0.8746 - mq_acc_11: 0.8748 - mq_acc_12: 0.8750 - mq_acc_13: 0.8752 - mq_acc_14: 0.8755 - mq_acc_15: 0.8757 - mq_acc_16: 0.8760 - mq_acc_17: 0.8761 - mq_acc_18: 0.8766 - mq_acc_19: 0.8767 - mq_acc_20: 0.8768 - mq_acc_1s_0: 0.6869 - mq_acc_1s_1: 0.6869 - mq_acc_1s_2: 0.6871 - mq_acc_1s_3: 0.6874 - mq_acc_1s_4: 0.6882 - mq_acc_1s_5: 0.6896 - mq_acc_1s_6: 0.6916 - mq_acc_1s_7: 0.6936 - mq_acc_1s_8: 0.6963 - mq_acc_1s_9: 0.6989 - mq_acc_1s_10: 0.7014 - mq_acc_1s_11: 0.7035 - mq_acc_1s_12: 0.7055 - mq_acc_1s_13: 0.7071 - mq_acc_1s_14: 0.7091 - mq_acc_1s_15: 0.7109 - mq_acc_1s_16: 0.7125 - mq_acc_1s_17: 0.7136 - mq_acc_1s_18: 0.7149 - mq_acc_1s_19: 0.7163 - mq_acc_1s_20: 0.7168 
0.05    mq_acc_0: 0.8699 - mq_acc_1: 0.8737 - mq_acc_2: 0.8737 - mq_acc_3: 0.8739 - mq_acc_4: 0.8743 - mq_acc_5: 0.8751 - mq_acc_6: 0.8763 - mq_acc_7: 0.8773 - mq_acc_8: 0.8785 - mq_acc_9: 0.8800 - mq_acc_10: 0.8811 - mq_acc_11: 0.8822 - mq_acc_12: 0.8829 - mq_acc_13: 0.8833 - mq_acc_14: 0.8839 - mq_acc_15: 0.8845 - mq_acc_16: 0.8846 - mq_acc_17: 0.8850 - mq_acc_18: 0.8852 - mq_acc_19: 0.8854 - mq_acc_20: 0.8857 - mq_acc_1s_0: 0.6869 - mq_acc_1s_1: 0.6870 - mq_acc_1s_2: 0.6876 - mq_acc_1s_3: 0.6900 - mq_acc_1s_4: 0.6944 - mq_acc_1s_5: 0.7008 - mq_acc_1s_6: 0.7075 - mq_acc_1s_7: 0.7139 - mq_acc_1s_8: 0.7194 - mq_acc_1s_9: 0.7237 - mq_acc_1s_10: 0.7271 - mq_acc_1s_11: 0.7302 - mq_acc_1s_12: 0.7322 - mq_acc_1s_13: 0.7349 - mq_acc_1s_14: 0.7357 - mq_acc_1s_15: 0.7365 - mq_acc_1s_16: 0.7374 - mq_acc_1s_17: 0.7378 - mq_acc_1s_18: 0.7377 - mq_acc_1s_19: 0.7377 - mq_acc_1s_20: 0.7376
0.025   mq_acc_0: 0.8699 - mq_acc_1: 0.8741 - mq_acc_2: 0.8742 - mq_acc_3: 0.8749 - mq_acc_4: 0.8764 - mq_acc_5: 0.8782 - mq_acc_6: 0.8798 - mq_acc_7: 0.8809 - mq_acc_8: 0.8819 - mq_acc_9: 0.8830 - mq_acc_10: 0.8839 - mq_acc_11: 0.8844 - mq_acc_12: 0.8849 - mq_acc_13: 0.8852 - mq_acc_14: 0.8852 - mq_acc_15: 0.8853 - mq_acc_16: 0.8854 - mq_acc_17: 0.8850 - mq_acc_18: 0.8849 - mq_acc_19: 0.8847 - mq_acc_20: 0.8846 - mq_acc_1s_0: 0.6869 - mq_acc_1s_1: 0.6870 - mq_acc_1s_2: 0.6895 - mq_acc_1s_3: 0.6946 - mq_acc_1s_4: 0.7031 - mq_acc_1s_5: 0.7107 - mq_acc_1s_6: 0.7170 - mq_acc_1s_7: 0.7223 - mq_acc_1s_8: 0.7270 - mq_acc_1s_9: 0.7308 - mq_acc_1s_10: 0.7330 - mq_acc_1s_11: 0.7349 - mq_acc_1s_12: 0.7362 - mq_acc_1s_13: 0.7365 - mq_acc_1s_14: 0.7373 - mq_acc_1s_15: 0.7381 - mq_acc_1s_16: 0.7381 - mq_acc_1s_17: 0.7382 - mq_acc_1s_18: 0.7379 - mq_acc_1s_19: 0.7377 - mq_acc_1s_20: 0.7374
0.01    mq_acc_0: 0.8699 - mq_acc_1: 0.8743 - mq_acc_2: 0.8747 - mq_acc_3: 0.8761 - mq_acc_4: 0.8776 - mq_acc_5: 0.8794 - mq_acc_6: 0.8807 - mq_acc_7: 0.8816 - mq_acc_8: 0.8829 - mq_acc_9: 0.8839 - mq_acc_10: 0.8842 - mq_acc_11: 0.8847 - mq_acc_12: 0.8850 - mq_acc_13: 0.8850 - mq_acc_14: 0.8848 - mq_acc_15: 0.8849 - mq_acc_16: 0.8845 - mq_acc_17: 0.8841 - mq_acc_18: 0.8840 - mq_acc_19: 0.8840 - mq_acc_20: 0.8837 - mq_acc_1s_0: 0.6869 - mq_acc_1s_1: 0.6874 - mq_acc_1s_2: 0.6912 - mq_acc_1s_3: 0.6983 - mq_acc_1s_4: 0.7070 - mq_acc_1s_5: 0.7137 - mq_acc_1s_6: 0.7195 - mq_acc_1s_7: 0.7251 - mq_acc_1s_8: 0.7294 - mq_acc_1s_9: 0.7321 - mq_acc_1s_10: 0.7341 - mq_acc_1s_11: 0.7352 - mq_acc_1s_12: 0.7364 - mq_acc_1s_13: 0.7368 - mq_acc_1s_14: 0.7376 - mq_acc_1s_15: 0.7382 - mq_acc_1s_16: 0.7383 - mq_acc_1s_17: 0.7379 - mq_acc_1s_18: 0.7378 - mq_acc_1s_19: 0.7377 - mq_acc_1s_20: 0.7374
0.0     mq_acc_0: 0.8699 - mq_acc_1: 0.8744 - mq_acc_2: 0.8751 - mq_acc_3: 0.8763 - mq_acc_4: 0.8785 - mq_acc_5: 0.8798 - mq_acc_6: 0.8809 - mq_acc_7: 0.8821 - mq_acc_8: 0.8833 - mq_acc_9: 0.8838 - mq_acc_10: 0.8843 - mq_acc_11: 0.8846 - mq_acc_12: 0.8845 - mq_acc_13: 0.8845 - mq_acc_14: 0.8843 - mq_acc_15: 0.8839 - mq_acc_16: 0.8836 - mq_acc_17: 0.8836 - mq_acc_18: 0.8835 - mq_acc_19: 0.8834 - mq_acc_20: 0.8834 - mq_acc_1s_0: 0.6869 - mq_acc_1s_1: 0.6881 - mq_acc_1s_2: 0.6924 - mq_acc_1s_3: 0.7002 - mq_acc_1s_4: 0.7084 - mq_acc_1s_5: 0.7150 - mq_acc_1s_6: 0.7211 - mq_acc_1s_7: 0.7262 - mq_acc_1s_8: 0.7300 - mq_acc_1s_9: 0.7325 - mq_acc_1s_10: 0.7344 - mq_acc_1s_11: 0.7356 - mq_acc_1s_12: 0.7366 - mq_acc_1s_13: 0.7369 - mq_acc_1s_14: 0.7380 - mq_acc_1s_15: 0.7380 - mq_acc_1s_16: 0.7383 - mq_acc_1s_17: 0.7379 - mq_acc_1s_18: 0.7379 - mq_acc_1s_19: 0.7376 - mq_acc_1s_20: 0.7374
'''
