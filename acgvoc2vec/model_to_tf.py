import sentence_transformers
import tensorflow as tf
from sentence_transformers import SentenceTransformer, models
from torch import nn
from transformers import AutoTokenizer, TFAutoModel
import keras.backend as K

def sentencetransformer_to_tensorflow(model_path: str) -> tf.keras.Model:
    """Convert SentenceTransformer model at model_path to TensorFlow Keras model"""
    # 1. Load the Transformer model
    tf_model = TFAutoModel.from_pretrained(model_path, from_pt=True)



    input = {}
    input_ids = tf.keras.Input(shape=(None,), name='input_ids', dtype=tf.int32)
    attention_mask = tf.keras.Input(shape=(None,), name='attention_mask', dtype=tf.int32)
    input['input_ids'] = input_ids
    input['attention_mask'] = attention_mask
    # token_type_ids = tf.keras.Input(shape=(None,), dtype=tf.int32)

    # 2. Get the Hidden State
    output=tf_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # token_type_ids=token_type_ids,
    )
    # print(dir(tf_model))
    # print(tf_model.summary())
    # print(tf_model.outputs)
    # print(tf_model.output)
    # print(type(output))
    # print(dir(output))
    # print(list(output.hidden_states))
    # print( tf.squeeze(tf.stack(output.hidden_states,axis=-2),axis=0))
    hidden_state = output.last_hidden_state

    # 3. Mean pooling
    mean_pool = tf.keras.layers.GlobalAveragePooling1D()(
        inputs= hidden_state,mask=attention_mask
    )
    #print(mean_pool)
    # 4 nor
    #l2=tf.keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=1))


    # 4. Dense layer
    sentence_transformer_model = SentenceTransformer(model_path, device="cpu")
    dense_layer = sentence_transformer_model[-1]
    dense = pytorch_to_tensorflow_dense_layer(dense_layer)(mean_pool)
    #print(dense)
    # Return the model
    # model = tf.keras.Model(
    #     dict(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #       #  token_type_ids=token_type_ids,
    #     ),
    #     dense,
    # )


    model = tf.keras.Model(inputs=input, outputs=dense,

                           )

    return model


TORCH_TO_KERAS_ACTIVATION = {"torch.nn.modules.activation.Tanh": "tanh"}


def pytorch_to_tensorflow_dense_layer(dense_model: sentence_transformers.models.Dense) -> tf.keras.layers.Dense:
    weight = dense_model.linear.get_parameter("weight").cpu().detach().numpy().T
    bias = dense_model.linear.get_parameter("bias").cpu().detach().numpy()

    dense_config = dense_model.get_config_dict()

    return tf.keras.layers.Dense(
        dense_config["out_features"],
        input_shape=(dense_config["in_features"],),
        activation=TORCH_TO_KERAS_ACTIVATION[dense_config["activation_function"]],
        use_bias=dense_config["bias"],
        weights=[weight, bias],
    )


model_name = '/Volumes/Data/oysterqaq/Downloads/acg2vec_st_model-2023-03-03-20+30-epoch'
max_seq_length = 128
output_dimension = 512

word_embedding_model = models.Transformer(model_name,
                                          max_seq_length=max_seq_length)

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_cls_token=False,
                               pooling_mode_mean_tokens=True,
                               pooling_mode_max_tokens=False)

dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                           out_features=output_dimension,
                           activation_function=nn.Tanh())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

tf_model = sentencetransformer_to_tensorflow(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = [
    'やかん',
    'ケトル',
    'デッサン',
    '最古シリーズ',
    '静物',
    'pixiv最古絵',
    '伝説の始まり',
    'やかんタグ最古',
    'やかん5000users入り',
    '全てはここから始まった',
]

tf_tokens = dict(tokenizer(input_text, padding=True, truncation=True, return_tensors='tf'))
print(tf_tokens)

import numpy as np
a=tf_model(tf_tokens).numpy()
model=SentenceTransformer('/Volumes/Data/oysterqaq/Downloads/acg2vec_st_model-2023-03-03-20+30-epoch', device="cpu")

b=model.encode(input_text)

print(a[0])

print(b[0])
print(np.isclose(a,b
           ,
           atol=1e-6))
tf_model.save(model_name + "-tf")
