from __future__ import annotations
import os, sys, logging, re, math, requests
from typing import List
import torch, pandas as pd
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
BASE_DIR=os.getenv("BASE_DIR","/workspace/post_train/sql_agent")
MODEL_PATH=os.getenv("MODEL_PATH","/workspace/models/Qwen2.5-32B-Instruct")
OUTPUT_DIR=os.getenv("OUTPUT_DIR","/workspace/outputs/32B-LoRA-GRPO-TRL")
sys.path.insert(0,BASE_DIR)
LORA_R=int(os.getenv("LORA_R","8"))
LORA_ALPHA=int(os.getenv("LORA_ALPHA","16"))
MAX_PROMPT_LEN=int(os.getenv("MAX_PROMPT_LEN","2048"))
MAX_RESP_LEN=int(os.getenv("MAX_RESPONSE_LEN","512"))
N_ROLLOUTS=int(os.getenv("N_ROLLOUTS","2"))
LR=float(os.getenv("LR","1e-5"))
WARMUP_RATIO=float(os.getenv("WARMUP_RATIO","0.03"))
MAX_STEPS=int(os.getenv("MAX_STEPS","20"))
SAVE_STEPS=int(os.getenv("SAVE_STEPS","10"))
GRAD_ACCUM=int(os.getenv("GRAD_ACCUM","4"))
REWARD_MODEL_URL=os.getenv("REWARD_MODEL_URL","http://117.50.48.176:8400/score")
ENABLE_RM=os.getenv("ENABLE_RM_REWARD","1")=="1"
RM_WEIGHT=float(os.getenv("RM_WEIGHT","0.5"))
TARGET_MODULES=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
os.makedirs(OUTPUT_DIR,exist_ok=True)
LOCAL_RANK=int(os.environ.get("LOCAL_RANK",0))
IS_MAIN=(LOCAL_RANK==0)
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
log=logging.getLogger(__name__)
try:
 from sql_agent_v18 import sql_agent_prompt as SYSTEM_PROMPT
except:
 SYSTEM_PROMPT='You are an EBM assistant. Answer with PICO structure. Cite with [^id].'
try:
 from sql_agent_v18 import compute_format_reward as _FMT
 _format_reward=lambda a:_FMT(a,pico_filled=3)
except:
 def _format_reward(a):
  s=0.0
  for k in ['background','method','result','conclusion','intervention','population','outcome']:
   if re.search(k,a,re.IGNORECASE):s+=0.12
  return min(s,1.0)
_rm_s=requests.Session()
_RM_UP=None
def _probe_rm():
 global _RM_UP
 if not ENABLE_RM:
  _RM_UP=False;return
 try:
  _rm_s.post(REWARD_MODEL_URL,json={"prompt":"t","response":"t"},timeout=5)
  _RM_UP=True
 except:
  _RM_UP=False
 if IS_MAIN:log.info(f"RM available:{_RM_UP}")
def _rm_score(q,a):
 try:
  r=_rm_s.post(REWARD_MODEL_URL,json={"prompt":q,"response":a},timeout=10)
  if r.status_code==200:
   raw=float(r.json().get("score",0.0))
   z=(raw-(-3.4))/max(8.0,1e-6)
   return 1.0/(1.0+math.exp(-z))
 except:pass
 return 0.5
def reward_fn(completions,**kw):
 if _RM_UP is None:_probe_rm()
 qs=kw.get("question",[""]*len(completions))
 out=[]
 for c,q in zip(completions,qs):
  # TRL may pass str or list-of-dicts; extract text
  if isinstance(c,list):
   a=" ".join(m.get("content","") for m in c if isinstance(m,dict)).strip()
  else:
   a=str(c).strip()
  if not a:out.append(-0.5);continue
  fmt=_format_reward(a)
  if _RM_UP:
   rm=_rm_score(q,a)
   t=(1-RM_WEIGHT)*fmt+RM_WEIGHT*rm
  else:t=fmt
  out.append(float(max(-2.0,min(2.0,t))))
 if IS_MAIN:log.info(f"[reward] mean={sum(out)/len(out):.3f}")
 return out
def build_dataset(split="train"):
 path=os.path.join(BASE_DIR,"data",f"{split}.parquet")
 df=pd.read_parquet(path)
 recs=[]
 for _,row in df.iterrows():
  q=str(row.get("question","")).strip()
  if not q:continue
  recs.append({"prompt":[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}],"question":q})
 if IS_MAIN:log.info(f"Built {len(recs)} prompts")
 return Dataset.from_list(recs)
def main():
 if IS_MAIN:
  log.info(f"=== TRL GRPO 32B model={MODEL_PATH} steps={MAX_STEPS} lr={LR} ===")
 # Set WandB credentials
 os.environ.setdefault("WANDB_BASE_URL", "http://103.139.212.228:3005")
 os.environ.setdefault("WANDB_API_KEY", "local-f2ca8cd44276ac92ca0a2c12641a6902beb6847d")
 os.environ.setdefault("WANDB_PROJECT", "32b-lora-grpo-trl")
 _probe_rm()
 ds=build_dataset("train")
 lora=LoraConfig(r=LORA_R,lora_alpha=LORA_ALPHA,lora_dropout=0.05,target_modules=TARGET_MODULES,bias="none",task_type="CAUSAL_LM")
 cfg=GRPOConfig(
  output_dir=OUTPUT_DIR,max_steps=MAX_STEPS,
  per_device_train_batch_size=1,gradient_accumulation_steps=GRAD_ACCUM,
  learning_rate=LR,warmup_ratio=WARMUP_RATIO,lr_scheduler_type="cosine",
  num_generations=N_ROLLOUTS,max_completion_length=MAX_RESP_LEN,
  save_steps=SAVE_STEPS,logging_steps=1,bf16=True,
  gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
  dataloader_num_workers=0,report_to=["wandb"],run_name=f"qwen32b-lora-r{LORA_R}-trl",
  beta=0.1,epsilon=0.2,loss_type="grpo",
 )
 if IS_MAIN:log.info("Loading model (QLoRA 4-bit)...")
 bnb_cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4")
 model=AutoModelForCausalLM.from_pretrained(MODEL_PATH,quantization_config=bnb_cfg,device_map={"":LOCAL_RANK},trust_remote_code=True,torch_dtype=torch.bfloat16)
 t=GRPOTrainer(model=model,reward_funcs=reward_fn,args=cfg,train_dataset=ds,peft_config=lora)
 if IS_MAIN:log.info("Training...")
 t.train()
 if IS_MAIN:
  t.save_model(OUTPUT_DIR)
  log.info(f"Done. Saved to {OUTPUT_DIR}")
if __name__=="__main__":main()
