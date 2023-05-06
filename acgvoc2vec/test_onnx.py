import onnx
import onnxruntime as rt
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

onnx_model = onnx.load('/Volumes/Data/oysterqaq/PycharmProjects/ACG2vec-model/acgvoc2vec/onnx_model.onx')
onnx.checker.check_model(onnx_model)
print('The model is checked!')
opt = rt.SessionOptions()
opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
opt.log_severity_level = 3
opt.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

sess = rt.InferenceSession('/Volumes/Data/oysterqaq/PycharmProjects/ACG2vec-model/acgvoc2vec/onnx_model.onx', opt) # Loads the model


sentences = ['やかん',
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
modelA = SentenceTransformer('/Volumes/Data/oysterqaq/Downloads/acg2vec_st_model-2023-03-03-20+30-epoch')
print(modelA.tokenize(sentences))
tokenizer = AutoTokenizer.from_pretrained('/Volumes/Data/oysterqaq/Downloads/acg2vec_st_model-2023-03-03-20+30-epoch')
req=[]
for s in sentences:
    b = tokenizer(s)
    b = dict(b)
    req.append(b)
onnx_result = sess.run(None,[ dict( {'input_ids': [101, 24109, 20572, 10477, 4458, 2747, 6223, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]})])
print(onnx_result)
