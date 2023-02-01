import pandas as pd
import sqlalchemy
from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer
from sentence_transformers import losses,evaluation
from torch.utils.data import DataLoader

def test(sentences):
    embeddings = model.encode(sentences)
    key_value ={}
    for i in range(len(embeddings)):
        key_value[sentences[i]]=sentence_transformers.util.cos_sim(embeddings[0],embeddings[i])
    score=sorted(key_value.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    for each in score:
        print(each, end = ',\n')





model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# model = SentenceTransformer('./acg2vec_st_model_ck')


# load dataset
engine = sqlalchemy.create_engine('mysql+pymysql://root:Cheerfun.dev@local.ipv4.host:3306/deepix?charset=utf8')
sql = "select sentence_1,sentence_2 from fine_tune_st_dataset"
data_from_db = pd.read_sql(sql, engine)
dataset = []

for i in range(len(data_from_db.sentence_1)):
    if str.isspace(data_from_db.sentence_1[i]) or  str.isspace(data_from_db.sentence_2[i]):
        continue
    dataset.append(InputExample(texts=[data_from_db.sentence_1[i], data_from_db.sentence_2[i]]))
del data_from_db
batch_size = 240
train_dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

train_loss = losses.MultipleNegativesRankingLoss(model=model)
num_epochs = 20
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data

# train
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps, output_path='acg2vec_st_model',checkpoint_path='acg2vec_st_model_ck',use_amp=True,checkpoint_save_steps=6000,checkpoint_save_total_limit=10)

#
#
# sentences = ["亚丝娜","神奇宝贝","手冢治虫","欧尔麦特", "刀劍神域","刀剑神域","结城明日奈","結城明日奈","とある魔術の禁書目録","科學超電磁砲","桐人","mikoto misaka"
#              ,"MISAKA MIKOTO","アスナ","Sword Art Online","桐人","結城明日奈"
#             ,"莉法/桐谷直叶","诗乃","漩涡鸣人","佐助"]
# #,""
# #sentences = ["瞳の中の暗殺者", "警察官を狙った連続殺人事件が発生"]
#
# test(sentences)
# print("--------------")
# sentences = ["火影忍者","神奇宝贝","手冢治虫","欧尔麦特", "刀劍神域","刀剑神域","结城明日奈","結城明日奈","とある魔術の禁書目録","科學超電磁砲","桐人","mikoto misaka"
#              ,"MISAKA MIKOTO","アスナ","Sword Art Online","桐人","結城 明日奈"
#             ,"莉法/桐谷直叶","诗乃","漩涡鸣人","佐助"]
#
# test(sentences)
#
# print("--------------")
# sentences = ["神奇宝贝","宠物小精灵","小火龙","杰尼龟","水箭龟","呆呆兽","手冢治虫","欧尔麦特", "刀劍神域","刀剑神域","结城明日奈","結城明日奈","とある魔術の禁書目録","科學超電磁砲","桐人","mikoto misaka"
#              ,"MISAKA MIKOTO","アスナ","Sword Art Online","桐人","結城 明日奈"
#             ,"莉法/桐谷直叶","诗乃","漩涡鸣人","佐助"]
#
#
# test(sentences)
# print("--------------")
# sentences = ["数码宝贝","八神太一","大便兽","亚古兽","暴龙兽","天女兽","神奇宝贝","小火龙","杰尼龟","水箭龟","呆呆兽","手冢治虫","欧尔麦特", "刀劍神域","刀剑神域","结城明日奈","結城明日奈","とある魔術の禁書目録","科學超電磁砲","桐人","mikoto misaka"
#              ,"MISAKA MIKOTO","アスナ","Sword Art Online","桐人","結城 明日奈"
#             ,"莉法/桐谷直叶","诗乃","漩涡鸣人","佐助"]
#
# test(sentences)
#
# print("--------------")
# sentences = ["進撃の巨人","谏山创","艾伦·耶格尔","三笠·阿克曼","阿尔敏","进击的巨人","八神太一","大便兽","亚古兽","暴龙兽","天女兽","神奇宝贝","小火龙","杰尼龟","水箭龟","呆呆兽","手冢治虫","欧尔麦特", "刀劍神域","刀剑神域","结城明日奈","結城明日奈","とある魔術の禁書目録","科學超電磁砲","桐人","mikoto misaka"
#              ,"MISAKA MIKOTO","アスナ","Sword Art Online","桐人","結城 明日奈"
#             ,"莉法/桐谷直叶","诗乃","漩涡鸣人","佐助"]
#
# test(sentences)
#
#
#
# print("--------------")
# sentences = ["鬼灭之刃","灶门炭治郎","灶门祢豆子","水之呼吸","我妻善逸","嘴平伊之助","進撃の巨人","谏山创","艾伦·耶格尔","三笠·阿克曼","阿尔敏","进击的巨人","八神太一","大便兽","亚古兽","暴龙兽","天女兽","神奇宝贝","小火龙","杰尼龟","水箭龟","呆呆兽","手冢治虫","欧尔麦特", "刀劍神域","刀剑神域","结城明日奈","結城明日奈","とある魔術の禁書目録","科學超電磁砲","桐人","mikoto misaka"
#              ,"MISAKA MIKOTO","アスナ","Sword Art Online","桐人","結城 明日奈"
#             ,"莉法/桐谷直叶","诗乃","漩涡鸣人","佐助"]
#
# test(sentences)
#
#
#
#
# print("--------------")
# sentences = ["幸平创真","食戟のソーマ","薙切 えりな","田所 恵","タクミ・アルディーニ","鬼灭之刃","灶门炭治郎","灶门祢豆子","水之呼吸","我妻善逸","嘴平伊之助","進撃の巨人","谏山创","艾伦·耶格尔","三笠·阿克曼","阿尔敏","进击的巨人","八神太一","大便兽","亚古兽","暴龙兽","天女兽","神奇宝贝","小火龙","杰尼龟","水箭龟","呆呆兽","手冢治虫","欧尔麦特", "刀劍神域","刀剑神域","结城明日奈","結城明日奈","とある魔術の禁書目録","科學超電磁砲","桐人","mikoto misaka"
#              ,"MISAKA MIKOTO","アスナ","Sword Art Online","桐人","結城 明日奈"
#             ,"莉法/桐谷直叶","诗乃","漩涡鸣人","佐助"]
#
# test(sentences)
#
#
# print("--------------")
# sentences = ["命运石之门登场角色","冈部伦太郎","椎名真由理","牧濑红莉栖","食戟のソーマ","幸平创真","薙切 えりな","田所 恵","タクミ・アルディーニ","鬼灭之刃","灶门炭治郎","灶门祢豆子","水之呼吸","我妻善逸","嘴平伊之助","進撃の巨人","谏山创","艾伦·耶格尔","三笠·阿克曼","阿尔敏","进击的巨人","八神太一","大便兽","亚古兽","暴龙兽","天女兽","神奇宝贝","小火龙","杰尼龟","水箭龟","呆呆兽","手冢治虫","欧尔麦特", "刀劍神域","刀剑神域","结城明日奈","結城明日奈","とある魔術の禁書目録","科學超電磁砲","桐人","mikoto misaka"
#              ,"MISAKA MIKOTO","アスナ","Sword Art Online","桐人","結城 明日奈"
#             ,"莉法/桐谷直叶","诗乃","漩涡鸣人","佐助"]
#
# test(sentences)
#
#
#
#
# print("--------------")
# sentences = ["東京喰種トーキョーグール","金木 研","佐佐木琲世","霧嶋 董香","命运石之门登场角色","冈部伦太郎","椎名真由理","牧濑红莉栖","食戟のソーマ","幸平创真","薙切 えりな","田所 恵","タクミ・アルディーニ","鬼灭之刃","灶门炭治郎","灶门祢豆子","水之呼吸","我妻善逸","嘴平伊之助","進撃の巨人","谏山创","艾伦·耶格尔","三笠·阿克曼","阿尔敏","进击的巨人","八神太一","大便兽","亚古兽","暴龙兽","天女兽","神奇宝贝","小火龙","杰尼龟","水箭龟","呆呆兽","手冢治虫","欧尔麦特", "刀劍神域","刀剑神域","结城明日奈","結城明日奈","とある魔術の禁書目録","科學超電磁砲","桐人","mikoto misaka"
#              ,"MISAKA MIKOTO","アスナ","Sword Art Online","桐人","結城 明日奈"
#             ,"莉法/桐谷直叶","诗乃","漩涡鸣人","佐助"]
#
# test(sentences)
#
#
#
# print("--------------")
# sentences = ["科學超電磁砲","金木 研","佐佐木琲世","霧嶋 董香","命运石之门登场角色","冈部伦太郎","椎名真由理","牧濑红莉栖","食戟のソーマ","幸平创真","薙切 えりな","田所 恵","タクミ・アルディーニ","鬼灭之刃","灶门炭治郎","灶门祢豆子","水之呼吸","我妻善逸","嘴平伊之助","進撃の巨人","谏山创","艾伦·耶格尔","三笠·阿克曼","阿尔敏","进击的巨人","八神太一","大便兽","亚古兽","暴龙兽","天女兽","神奇宝贝","小火龙","杰尼龟","水箭龟","呆呆兽","手冢治虫","欧尔麦特", "刀劍神域","刀剑神域","结城明日奈","結城明日奈","とある魔術の禁書目録","科學超電磁砲","桐人","mikoto misaka"
#              ,"MISAKA MIKOTO","アスナ","Sword Art Online","桐人","結城 明日奈"
#             ,"莉法/桐谷直叶","诗乃","漩涡鸣人","佐助"]
#
# test(sentences)

#pytorch to tensorflow https://skeptric.com/sentencetransformers-to-tensorflow/





