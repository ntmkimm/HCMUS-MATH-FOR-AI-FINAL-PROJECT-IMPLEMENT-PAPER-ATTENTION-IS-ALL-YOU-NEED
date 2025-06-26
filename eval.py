from utils import *

'''
In ra các câu input và output và predict của file, eval trên toàn bộ dataset
'''

# CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 50
learning_rate = 3e-4
dropout = 0.2
n = 70 # số token tối đa
N = 4 # số identical layer của Encoder/Decoder
h = 4 # số head trong multihead attention
d_model = 256
d_ff = 1024

# BLEU Score: 0.6674 train2/ep10

input_language = 'en'
output_language = 'vi'

model_path = 'train2/model_epoch_10.pt'

# set up tokenizer
print("Setup tokenizer for english and vietnamese...")
tokenizer_input = setup_tokenizer(name='tokenizer_input.json')
tokenizer_output = setup_tokenizer(name='tokenizer_output.json')

# Tải mô hình đã lưu
model = setup_transformer(tokenizer_input.get_vocab_size(), tokenizer_output.get_vocab_size(), n, n, d_model, N, h, d_ff, dropout).to(device)
model.load_state_dict(torch.load(model_path))  # tên file mô hình của bạn
model.eval()  # Chuyển mô hình sang chế độ đánh giá

path = kagglehub.dataset_download("hungnm/englishvietnamese-translation")

print("Path to dataset files:", path)

en_sents = open(path + '/en_sents', "r").read().splitlines()
vi_sents = open(path + '/vi_sents', "r").read().splitlines()
raw_data = {
        "en": [line for line in en_sents],
        "vi": [line for line in vi_sents],
    }
df = pd.DataFrame(raw_data, columns=["en", "vi"])

print("Processing dataset...")
df = preprocessing(df[:])
print(df)
  
# build  val-set với max length n=70, tokenizer đã train trên tập train
print("Build val dataset...")
val_data = df[:].sample(frac=1, random_state=42).reset_index(drop=True)
val_data = build_dataset(val_data, n, tokenizer_input, tokenizer_output, input_col="en", output_col="vi")
val_data = TranslationDataset(val_data)

eval(model=model, data=val_data, tokenizer_input=tokenizer_input, tokenizer_output=tokenizer_output, n=n, device=device)