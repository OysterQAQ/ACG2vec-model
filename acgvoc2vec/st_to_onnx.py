import torch
import transformers
import numpy as np
from sentence_transformers import SentenceTransformer, models
TOKENIZERS_PARALLELISM=False
from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class OnnxEncoder:
    """OnxEncoder dedicated to run SentenceTransformer under OnnxRuntime."""

    def __init__(self, session, tokenizer, pooling, normalization):
        self.session = session
        self.tokenizer = tokenizer
        self.max_length = tokenizer.__dict__["model_max_length"]
        self.pooling = pooling
        self.normalization = normalization

    def encode(self, sentences: list):

        sentences = [sentences] if isinstance(sentences, str) else sentences

        inputs = {
            k: v.numpy()
            for k, v in self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).items()
        }

        hidden_state = self.session.run(None, inputs)
        print(len(hidden_state))
        print(len(hidden_state[0]))
        print(len(hidden_state[0][0]))
        print(len(hidden_state[0][0][0]))
        print(self.pooling.get_config_dict())
        print(type(self.pooling))
        sentence_embedding = self.pooling.forward(
            features={
                "token_embeddings": torch.Tensor(hidden_state[0]),
                "attention_mask": torch.Tensor(inputs.get("attention_mask")),
            },
        )
        print(sentence_embedding)

        if self.normalization is not None:
            sentence_embedding = self.normalization.forward(features=sentence_embedding)
        print(sentence_embedding["sentence_embedding"].shape)

        sentence_embedding = sentence_embedding["sentence_embedding"]

        if sentence_embedding.shape[0] == 1:
            sentence_embedding = sentence_embedding[0]


        return sentence_embedding.detach().numpy()


def sentence_transformers_onnx(
    model,
    path,
    do_lower_case=True,
    input_names=["input_ids", "attention_mask", "segment_ids"],
    providers=["CPUExecutionProvider"],
):
    """OnxRuntime for sentence transformers.

    Parameters
    ----------
    model
        SentenceTransformer model.
    path
        Model file dedicated to session inference.
    do_lower_case
        Either or not the model is cased.
    input_names
        Fields needed by the Transformer.
    providers
        Either run the model on CPU or GPU: ["CPUExecutionProvider", "CUDAExecutionProvider"].

    """
    try:
        import onnxruntime
    except:
        raise ValueError("You need to install onnxruntime.")

    model.save(path)

    configuration = transformers.AutoConfig.from_pretrained(
        path, from_tf=False, local_files_only=True
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        path,  from_tf=False, local_files_only=True
    )

    encoder = transformers.AutoModel.from_pretrained(
        path, from_tf=False, config=configuration, local_files_only=True
    )

    st = ["cherche","cherche",]

    inputs = tokenizer(
        st,
        padding=True,
        truncation=True,
        max_length=tokenizer.__dict__["model_max_length"],
        return_tensors="pt",
    )

    model.eval()

    with torch.no_grad():

        symbolic_names = {0: "batch_size", 1: "max_seq_len"}

        torch.onnx.export(
            encoder,
            args=tuple(inputs.values()),
            f=f"{path}.onx",
            opset_version=13,  # ONX version needs to be >= 13 for sentence transformers.
            do_constant_folding=True,
            input_names=input_names,
            output_names=["start", "end"],
            dynamic_axes={
                "input_ids": symbolic_names,
                "attention_mask": symbolic_names,
                #"segment_ids": symbolic_names,
                "start": symbolic_names,
                "end": symbolic_names,
            },
        )

        normalization = None
        for modules in model.modules():
            for idx, module in enumerate(modules):
                print(idx)
                print(module)
                if idx == 1:
                    pooling = module
                if idx == 2:
                    normalization = module
            break

        return OnnxEncoder(
            session=onnxruntime.InferenceSession(f"{path}.onx", providers=providers),
            tokenizer=tokenizer,
            pooling=pooling,
            normalization=normalization,
        )


model = sentence_transformers_onnx(
    model = SentenceTransformer("/Volumes/Data/oysterqaq/Downloads/acg2vec_st_model-2023-03-03-20+30-epoch"),
    path = "onnx_model",
    input_names=["input_ids", "attention_mask"]
)
sentences=['やかん',
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
a=model.encode(sentences)
print()

tokenizer = AutoTokenizer.from_pretrained("/Volumes/Data/oysterqaq/Downloads/acg2vec_st_model-2023-03-03-20+30-epoch")
modelA = SentenceTransformer('/Volumes/Data/oysterqaq/Downloads/acg2vec_st_model-2023-03-03-20+30-epoch', device="cpu")
b=modelA.encode(sentences)

batch = tokenizer(sentences, padding=True)
sentences=['やかん', ]
c=modelA.encode(sentences)
# print(a[2])
# print(b[2])
# print('-----------')
# print(model.encode([sentences[2]]))
# print(modelA.encode([sentences[2]]))

print(np.isclose(c,
          b[0],
           atol=1e-6).all())
