import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
import torch

# HUGGINGFACE & ADAKV EKLEMELERİ
from transformers import AutoTokenizer, AutoModelForCausalLM
from adaptive_snapkv.monkeypatch.monkeypatch import replace_llama_adaptive

template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
# Not: Sadece 0-shot template'ini bıraktım, eğer RAG veya COT kullanacaksan
# eski koddaki template yollarını buraya ekleyebilirsin.

def config_compress(model, window_size=32, base_capacity=1024, kernel_size=7, pooling="maxpool", floor_alpha=0.5, pyram_mode=True, beta=20):
    # AdaKV hiperparametre ayarları (b=20% için optimize)
    model.model.config.window_size = window_size
    model.model.config.base_capacity = base_capacity
    model.model.config.kernel_size = kernel_size
    model.model.config.pooling = pooling
    model.model.config.floor_alpha = floor_alpha
    model.model.config.pyram_mode = pyram_mode
    model.model.config.pyram_beta = beta
    return model

def query_llm_local(prompt, model, tokenizer, temperature=0.1, max_new_tokens=128):
    # Promptu token'a çevir ve GPU'ya yolla
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0.0 else None,
            do_sample=(temperature > 0.0),
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Sadece yeni üretilen tokenları al
    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_adakv_b{args.beta}.jsonl")
    
    # 1. MODEL VE TOKENIZER YÜKLEME (ADAKV YAMASI İLE)
    print("Model yükleniyor ve AdaKV yaması atılıyor...")
    replace_llama_adaptive() # Llama monkeypatch
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Sıkıştırma ayarlarını uygula
    model = config_compress(model, base_capacity=args.base_capacity, pyram_mode=True, beta=args.beta)
    model.eval()
    print("Model hazır!")

    # 2. VERİ SETİNİ YÜKLE
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    data_all = [{"_id": item["_id"], "domain": item["domain"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # Kaldığı yerden devam etme mantığı
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    
    fout = open(out_file, 'a', encoding='utf-8')
    
    # 3. TEK İŞLEMCİ İLE (OOM YEMEMEK İÇİN) SIRAYLA ÜRETİM YAP
    for item in tqdm(data_all):
        if item["_id"] in has_data:
            continue
            
        context = item['context']
        template = template_0shot
        
        # Promptu hazırla
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        
        # Modeli sorgula
        output = query_llm_local(prompt, model, tokenizer, temperature=0.1, max_new_tokens=128)
        
        if output == '':
            continue
            
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000] # Logu çok şişirmemek için kırptık
        
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--base_capacity", type=int, default=1024, help="b değerine göre kapasite ayarı")
    parser.add_argument("--beta", type=int, default=20, help="Tablodaki b yüzdesi (örn: 20)")
    args = parser.parse_args()
    main()