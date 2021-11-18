# imports
import torch
from transformers import GPT2Tokenizer
from model.modeling_gpt2 import GPT2HeadWithValueModel
from transformers import BartTokenizer
from transformers import BartForConditionalGeneration
from model.modeling_bart import BARTHeadWithValueModel, respond_to_batch
#from trl.ppo import PPOTrainer
from model.ppo import PPOTrainer
from rouge import Rouge
import numpy as np
import os
'''
# get models
gpt2_model = GPT2HeadWithValueModel.from_pretrained('facebook/bart-large-cnn')
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('facebook/bart-large-cnn')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('facebook/bart-large-cnn')

# initialize trainer
ppo_config = {'batch_size': 1, 'forward_batch_size': 1}
ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **ppo_config)

# encode a query
query_txt = "This morning I went to the "
query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")

# get model response
response_tensor  = respond_to_batch(gpt2_model, query_tensor)
response_txt = gpt2_tokenizer.decode(response_tensor[0,:])

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = torch.tensor([1.0])

# train model with ppo
train_stats = ppo_trainer.step(query_tensor, response_tensor, reward)
'''

def main():
    epochs = 50
    rouge = Rouge()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    with open('/home/ypd-19-2/Prompt/model_input_files_zero_shot/train_as_test/train.source', 'r') as r:
        source_lines = r.readlines()
    with open('/home/ypd-19-2/Prompt/model_input_files_zero_shot/train_as_test/train.target', 'r') as r:
        target_lines = r.readlines()

    # get models

    '''
    gpt2_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').cuda()
    gpt2_model_ref = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').cuda()
    gpt2_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    '''
    gpt2_model = BARTHeadWithValueModel.from_pretrained('facebook/bart-large-cnn').cuda(device="cuda:0")
    gpt2_model_ref = BARTHeadWithValueModel.from_pretrained('facebook/bart-large-cnn').cuda(device="cuda:1")
    gpt2_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    '''
    gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2').cuda()
    gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2').cuda()
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    '''




    # initialize trainer
    ppo_config = {'batch_size': 1, 'forward_batch_size': 1, 'lr': 7.5e-8}
    ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **ppo_config)
    index = 0
    rouge_list = []
    for epoch in range(epochs) :
        for s, t in zip(source_lines, target_lines):
            query_tensor = gpt2_tokenizer.encode(s, max_length=1024, return_tensors="pt", truncation=True).cuda()
            #query_tensor = gpt2_tokenizer([s], max_length=1024, return_tensors='pt')

            # get model response
            response_tensor = respond_to_batch(gpt2_model, query_tensor)
            #response_txt = [gpt2_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in response_tensor][0]
            response_txt = gpt2_tokenizer.decode(response_tensor[0,:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            #print(response_txt)
            rouge_score = rouge.get_scores(response_txt, t)[0]['rouge-l']['f']
            rouge_list.append(rouge_score)
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = torch.tensor([rouge_score]).cuda()
            # train model with ppo
            train_stats = ppo_trainer.step(query_tensor, response_tensor, reward)
            index = index + 1
            if index % 100 == 0:
                print('rouge_score:', np.mean(rouge_list))
                rouge_list = []
            #print("bupt")

if __name__ == '__main__':
    main()