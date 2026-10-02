import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def run_inference(prompt: str):
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_dir = "./results"  # โฟลเดอร์เก็บ Adapter ที่ได้จากการ Fine-tune

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # โหลด LoRA Adapter เข้ากับ Base Model (ถ้ามี Adapter แล้ว)
    try:
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        print("Successfully loaded fine-tuned LoRA adapters.")
    except Exception:
        print("No fine-tuned weights found in './results'. Running base model instead.")
        model = base_model

    # ฟอร์แมตระบบการคุยแบบ Chat Format ของ Qwen
    messages = [
        {"role": "system", "content": "You are a helpful local AI agent assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(base_model.device)

    print("\nGenerating response...\n")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
    )
    
    response = tokenizer.batch_decode(
        [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)],
        skip_special_tokens=True
    )[0]

    print("--- Response ---")
    print(response)

if __name__ == "__main__":
    test_prompt = "Hello! Introduce yourself briefly."
    run_inference(test_prompt)