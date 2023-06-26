import tensorflow as tf
from keras import optimizers
from keras.optimizers import optimizer
from keras.engine import data_adapter

class PCGradWarp(tf.keras.Model):


    def compute_gradients(self, loss, var_list):
        num_tasks=len(loss)
        #堆叠方便向量化并行
        loss = tf.stack(loss)
        #打乱任务顺序
        tf.random.shuffle(loss)
        #计算每个任务的梯度并且打平
        grads_per_task = tf.vectorized_map(lambda x: tf.concat([tf.reshape(grad, [-1, ])
                                                            for grad in tf.gradients(x, var_list)
                                                            if grad is not None], axis=0), loss)

        # Compute gradient projections.
        def proj_grad(grad_task):
            for k in range(num_tasks):
                #计算所有任务与内积
                inner_product = tf.reduce_sum(grad_task * grads_per_task[k])
                #计算与所有任务cos相似度
                proj_direction = inner_product / tf.reduce_sum(grads_per_task[k] * grads_per_task[k])
                #投影
                grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_per_task[k]
            return grad_task

        proj_grads_flatten = tf.vectorized_map(proj_grad, grads_per_task)
        proj_grads = []
        for j in range(num_tasks):
            start_idx = 0
            for idx, var in enumerate(var_list):
                grad_shape = var.get_shape()
                flatten_dim = tf.math.reduce_prod([grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
                proj_grad = proj_grads_flatten[j][start_idx:start_idx + flatten_dim]
                proj_grad = tf.reshape(proj_grad, grad_shape)
                if len(proj_grads) < len(var_list):
                    proj_grads.append(proj_grad)
                else:
                    proj_grads[idx] += proj_grad
                start_idx += flatten_dim
        grads_and_vars = list(zip(proj_grads, var_list))
        return grads_and_vars


    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)


    def train_step(self, data):

        # x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # # Run forward pass.
        # with tf.GradientTape() as tape:
        #     y_pred = self(x, training=True)
        #     loss = self.compute_loss(x, y, y_pred, sample_weight)
        # self._validate_target_and_loss(y, loss)
        # # Run backwards pass.
        # self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        # return self.compute_metrics(x, y, y_pred, sample_weight)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)

        var_list = self.trainable_variables

        grads_and_vars =self.compute_gradients(loss.op.inputs, var_list)
        self.optimizer.apply_gradients(grads_and_vars)
        # print(grads_and_vars)




        # for perloss in loss.op.inputs:
        #     self._validate_target_and_loss(y, perloss)
        #     grads_and_vars = self.optimizer.compute_gradients(perloss, var_list, tape)
        #     self.optimizer.apply_gradients(grads_and_vars)

        # print(loss)
        # print(loss.op.inputs[0])

        #单独计算梯度


        # grads_and_vars = self.optimizer.compute_gradients(loss.op.inputs[0], var_list)

        #self.optimizer.apply_gradients(grads_and_vars)
        # self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        #self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        #self.optimizer.minimize(loss.op.inputs[0], self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)


