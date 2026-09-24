"""Competition inference using the EXISTING saved Qwen2.5-0.5B LoRA adapter at 1024 tokens.

No training occurs here. This is an explicitly authorized experimental Kaggle
submission. The notebook stays private, Internet-off, and writes only submission.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.baseline import TARGETS, normalized_frame
from src.finetune_lora import render_pair


def predict_batches(model, tokenizer, frame, device="cuda:0", max_length=1024, batch_size=2):
    import torch
    if len(frame)==0:
        raise ValueError("Test frame is empty")
    model.eval()
    out=[]
    with torch.inference_mode():
        for start in range(0,len(frame),batch_size):
            chunk=frame.iloc[start:start+batch_size]
            texts=[render_pair(row) for row in chunk.to_dict("records")]
            enc=tokenizer(texts,truncation=True,max_length=max_length,padding=True,return_tensors="pt")
            batch={k:v.to(device) for k,v in enc.items()}
            logits=model(**batch).logits.detach().float().cpu().numpy()
            if logits.ndim!=2 or logits.shape[1]!=3:
                raise ValueError(f"Expected Nx3 logits, got {logits.shape}")
            out.append(softmax(logits.astype(np.float64),axis=1))
    probs=np.vstack(out)
    probs=np.clip(probs,1e-12,1.0)
    return probs/probs.sum(axis=1,keepdims=True)


def run(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    base_dir=Path(args.base_model)
    adapter_dir=Path(args.adapter)
    if not (base_dir/"config.json").is_file():
        raise FileNotFoundError("Missing Qwen base config")
    if not (adapter_dir/"adapter_config.json").is_file():
        raise FileNotFoundError("Missing saved adapter config")
    if not (adapter_dir/"adapter_model.safetensors").is_file():
        raise FileNotFoundError("Missing saved adapter weights")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is required for this code submission")

    test=pd.read_csv(args.test)
    if "id" not in test.columns:
        raise ValueError("Competition test.csv requires id")
    frame=normalized_frame(test)

    tok=AutoTokenizer.from_pretrained(base_dir,local_files_only=True,trust_remote_code=False)
    if tok.pad_token_id is None:
        if tok.eos_token is None:
            raise ValueError("Tokenizer lacks pad and EOS")
        tok.pad_token=tok.eos_token
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base=AutoModelForSequenceClassification.from_pretrained(
        base_dir,num_labels=3,torch_dtype=dtype,local_files_only=True,trust_remote_code=False
    )
    base.config.pad_token_id=tok.pad_token_id
    base.config.use_cache=False
    model=PeftModel.from_pretrained(base,adapter_dir,is_trainable=False,local_files_only=True)
    model.to("cuda:0")
    probs=predict_batches(model,tok,frame,max_length=args.max_length,batch_size=args.batch_size)
    sub=pd.DataFrame(probs,columns=TARGETS)
    sub.insert(0,"id",test["id"])
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    sub.to_csv(out,index=False)
    print(f"Wrote {len(sub)} rows to {out}")
    print("Context tokens:",args.max_length,"batch size:",args.batch_size)
    return len(sub)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test",required=True)
    p.add_argument("--base-model",required=True)
    p.add_argument("--adapter",required=True)
    p.add_argument("--output",default="/kaggle/working/submission.csv")
    p.add_argument("--max-length",type=int,default=1024)
    p.add_argument("--batch-size",type=int,default=2)
    run(p.parse_args())
