from utils import *

import kagglehub

# CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 50
learning_rate = 1e-4

dropout = 0.2
n = 70 # số token tối đa
N = 4 # số identical layer của Encoder/Decoder
h = 4 # số head trong multihead attention
d_model = 512
d_ff = 2048

input_language = 'en'
output_language = 'vi'

# preprocess dataset
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
df = preprocessing(df)
print(df)

print("Setup tokenizer for english and vietnamese...")
tokenizer_input = setup_tokenizer(df["en"], min_frequency=2, name='tokenizer_input.json')
tokenizer_output = setup_tokenizer(df["vi"], min_frequency=2, name='tokenizer_output.json')

# build toàn bộ train-set, val-set với max length n=70, tokenizer đã train trên tập train
print("Build dataset...")
shuffled_df = df[:].sample(frac=1, random_state=42).reset_index(drop=True)

# Tính số lượng mẫu cho train và val
train_size = int(0.9 * len(shuffled_df))
val_size = len(shuffled_df) - train_size

# Chia thành train và val
train_data = shuffled_df[:train_size]
val_data = shuffled_df[train_size:]

train_data = build_dataset(train_data, n, tokenizer_input, tokenizer_output, input_col="en", output_col="vi")
val_data = build_dataset(val_data, n, tokenizer_input, tokenizer_output, input_col="en", output_col="vi")
train_data = TranslationDataset(train_data)
val_data = TranslationDataset(val_data)

# setup mô hình transformer
print("Setup Transformer...")
model = setup_transformer(tokenizer_input.get_vocab_size(), tokenizer_output.get_vocab_size(), n, n, d_model, N, h, d_ff, dropout).to(device)

# Optimizer và loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_output.token_to_id("<PAD>"), label_smoothing=0.1).to(device)

print("Training...")
train(model, train_data, val_data, tokenizer_input, tokenizer_output, optimizer, criterion, n, batch_size, epochs, device, save_dir="train1")