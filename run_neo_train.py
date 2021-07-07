import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPTNeoForCausalLM
import os

class ShowDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

def main():
    torch.manual_seed(42)
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M", bos_token='<|startoftext|>',
                                            eos_token='<|endoftext|>', pad_token='<|pad|>')
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")#.cuda()
    model.resize_token_embeddings(len(tokenizer))
    
    file_names = [name for name in os.listdir("friends") if name.endswith(".txt")]

    def read_script(file_name):
        txt_file = open(f"friends/{file_name}")
        script = txt_file.readlines()
        script_cleaned= [line[0:-2] for line in script if line!='\n'][0:25]
        text_string = " ".join(script_cleaned)
        txt_file.close()
        return text_string
    scripts = [read_script(file_name) for file_name in file_names]
    max_length = max([len(tokenizer.encode(script)) for script in scripts])

    dataset = ShowDataset(scripts, tokenizer, max_length=max_length)
    train_size = int(0.9 * len(dataset))

    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    training_args = TrainingArguments(output_dir='./results', num_train_epochs=5, logging_steps=5000, save_steps=5000,
                                    per_device_train_batch_size=2, per_device_eval_batch_size=2,
                                    warmup_steps=100, weight_decay=0.01, logging_dir='./logs')

    Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                'attention_mask': torch.stack([f[1] for f in data]),
                                                                'labels': torch.stack([f[0] for f in data])}).train()

    generated = tokenizer("<|startoftext|> ", return_tensors="pt").input_ids#.cuda()
    sample_outputs = model.generate(generated, do_sample=True, top_k=50, 
                                    max_length=2000, top_p=0.95, temperature=0.5, num_return_sequences=20)
    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

if __name__ == "__main__":
    main()
