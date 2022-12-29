from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import shutil
import os
from tqdm import tqdm, trange
import argparse

from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW

import cbert_utils_rf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_ids_to_str(ids, tokenizer):
    """converts token_ids into str."""
    tokens = []
    for token_id in ids:
        token = tokenizer._convert_id_to_token(token_id)
        tokens.append(token)
    outputs = cbert_utils_rf.rev_wordpiece(tokens)
    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="subj",type=str,
                        help="The name of the task to train.")

    AugProcessor = cbert_utils_rf.AugProcessor()   
    processors = {
    ## you can add your processor here
    "nsmc": AugProcessor,
    "korean-hate-speech-detection": AugProcessor,
}
    args = parser.parse_args()
    
    task_name = args.task_name
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]
    # label_list = processor.get_labels(task_name)
    
    tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base')
    
    def load_model(model_name):
        weights_path = os.path.join(model_name)
        model = torch.load(weights_path)
        return model
    
    # data_dir = os.path.join(data_dir, task_name)
    # output_dir = os.path.join(output_dir, task_name)
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    # shutil.copytree("aug_data/{}".format(task_name), output_dir)
    
    sentences = ['(현재 호텔주인 심정) 아18 난 마른하늘에 날벼락맞고 호텔망하게생겼는데 누군 계속 추모받네....',
                        '....한국적인 미인의 대표적인 분...너무나 곱고아름다운모습...그모습뒤의 슬픔을 미처 알지못했네요ㅠ',
                        '...못된 넘들...남의 고통을 즐겼던 넘들..이젠 마땅한 처벌을 받아야지..,그래야, 공정한 사회지...심은대로 거두거라...',
                        '1,2화 어설펐는데 3,4화 지나서부터는 갈수록 너무 재밌던데',
                        '1. 사람 얼굴 손톱으로 긁은것은 인격살해이고2. 동영상이 몰카냐? 메걸리안들 생각이 없노',
                        '10+8 진짜 이승기랑 비교된다',
                        '100년안에 남녀간 성전쟁 한번 크게 치룬 후 일부다처제, 여성의 정치참여 금지, 여성 투표권 삭제가 세계의 공통문화로 자리잡을듯. 암탉이 너무 울어댐.',
                        '10년뒤 윤서인은 분명히 재평가될것임. 말하나하나가 틀린게없음']
    
    
    label = ['0', '1', '0', '1', '0', '1', '0', '1']
    label_list = ['0', '1', '2']
    
    
    
    train_features, num_train_steps, train_dataloader = \
        cbert_utils_rf.construct_train_dataloader(train_examples=sentences, label_list=label_list, labels=label, max_seq_length=64, train_batch_size=32, num_train_epochs=9.0, tokenizer=tokenizer, device=device)
        
    # save_model_dir = os.path.join(args.save_model_dir, task_name)
    # if not os.path.exists(save_model_dir):
    #     os.mkdir(save_model_dir)
    MASK_id = cbert_utils_rf.convert_tokens_to_ids(['[MASK]'], tokenizer)[0]
    
    # origin_train_path = os.path.join(output_dir, "train_origin.tsv")
    # save_train_path = os.path.join(output_dir, "train.tsv")
    # shutil.copy(origin_train_path, save_train_path)
    
    # for e in trange(int(num_train_epochs=10), desc="Epoch"):
    for e in trange(10, desc="Epoch"):
        torch.cuda.empty_cache()
        # cbert_name = "{}/BertForMaskedLM_{}_epoch_{}".format(task_name.lower(), task_name.lower(), e+1) # e+1
        cbert_name = "/home/sangmin/cbert_aug-crayon/cbert_model/korean-hate-speech-detection/BertForMaskedLM_korean-hate-speech-detection_epoch_10" # e+1
        model = load_model(cbert_name)
        model.cuda()
        # shutil.copy(origin_train_path, save_train_path)
        # save_train_file = open(save_train_path, 'a')
        
        # tsv_writer = csv.writer(save_train_file, delimiter='\t')
        print('test1')
        for _, batch in enumerate(train_dataloader):
            model.eval()
            batch = tuple(t.cuda() for t in batch)
            init_ids, _, input_mask, segment_ids, _ = batch
            input_lens = [sum(mask).item() for mask in input_mask]
            # masked_idx = np.squeeze([np.random.randint(0, l, max(l//sample_ratio, 1)) for l in input_lens])
            masked_idx = np.squeeze([np.random.randint(0, l, max(l//7, 1)) for l in input_lens])
            for ids, idx in zip(init_ids, masked_idx):
                ids[idx] = MASK_id
            predictions = model(init_ids, input_mask, segment_ids)
            # predictions = torch.nn.functional.softmax(predictions[0]/temp, dim=2)            
            predictions = torch.nn.functional.softmax(predictions[0]/3.0, dim=2)
            for ids, idx, preds, seg in zip(init_ids, masked_idx, predictions, segment_ids):
                preds = torch.multinomial(preds, 1, replacement=True)[idx]
                if len(preds.size()) == 2:
                    preds = torch.transpose(preds, 0, 1)
                for pred in preds:
                    ids[idx] = pred
                    new_str = convert_ids_to_str(ids.cpu().numpy(), tokenizer)

                    print([new_str, seg[0].item()])
                    
            torch.cuda.empty_cache()
            
        predictions = predictions.detach().cpu()
        model.cpu()
        torch.cuda.empty_cache()
        # bak_train_path = os.path.join(output_dir, "train_epoch_{}.tsv".format(e))
        # shutil.copy(save_train_path, bak_train_path)

        
if __name__ == "__main__":
    main()
