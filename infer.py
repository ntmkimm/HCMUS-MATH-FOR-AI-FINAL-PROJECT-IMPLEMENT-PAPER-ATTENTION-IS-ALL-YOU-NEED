from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dropout = 0.2
n = 70 # số token tối đa
N = 4 # số identical layer của Encoder/Decoder
h = 4 # số head trong multihead attention
d_model = 256
d_ff = 1024

model_path = 'train2/model_epoch_10.pt'

input_language = 'en'
output_language = 'vi'

# set up tokenizer
print("Setup tokenizer for english and vietnamese...")
tokenizer_input = setup_tokenizer(name='tokenizer_input.json')
tokenizer_output = setup_tokenizer(name='tokenizer_output.json')

# Tải mô hình đã lưu
model = setup_transformer(tokenizer_input.get_vocab_size(), tokenizer_output.get_vocab_size(), n, n, d_model, N, h, d_ff, dropout).to(device)
model.load_state_dict(torch.load(model_path))  # tên file mô hình 
model.eval()  # Chuyển mô hình sang chế độ đánh giá

while True:
  print("Nhập input: ")
  sentence = input()
  if (sentence == 'exit'): break
  predicted_translation = predict(model, n, tokenizer_input, tokenizer_output, sentence, device)
  print("Output: ", predicted_translation, "\n")