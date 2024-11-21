# ************************ 模型部分代碼 ************************

from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import jieba
import string
import textwrap

device = "cuda"

# 加載模型和分詞器
peft_model_id = "CMLM/ZhongJing-2-1_8b"
base_model_id = "Qwen/Qwen1.5-1.8B-Chat"
model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
model.load_adapter(peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(
	peft_model_id, padding_side="right", trust_remote_code=True, pad_token=""
)


def get_model_response(question):
	# Create the prompt without context
	prompt = f"Question: {question}"
	messages = [
		# {"role": "system", "content": "You are a helpful medical assistant."},
		{"role": "system", "content": "You are a helpful Traditional Chinese Medicine Assistant."},
		{"role": "user", "content": prompt},
	]

	# Prepare the input
	text = tokenizer.apply_chat_template(
		messages, tokenize=False, add_generation_prompt=True
	)
	model_inputs = tokenizer([text], return_tensors="pt").to(device)

	# Generate the response
	generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
	generated_ids = [
		output_ids[len(input_ids) :]
		for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
	]

	# Decode the response
	response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
	return response


def predictor(texts):
	results = []
	for text in texts:
		# 創建消息格式
		messages = [
			{"role": "system", "content": "You are a helpful medical assistant."},
			{"role": "user", "content": text},
		]

		# 準備輸入
		formatted_text = tokenizer.apply_chat_template(
			messages, tokenize=False, add_generation_prompt=True
		)
		inputs = tokenizer([formatted_text], return_tensors="pt").to(model.device)

		with torch.no_grad():
			outputs = model(**inputs)
			logits = outputs.logits
			probs = torch.softmax(logits[0, -1], dim=-1)
			top_probs = torch.topk(probs, 2)
			prob_array = np.array(
				[top_probs.values[0].item(), top_probs.values[1].item()]
			)
			prob_array = prob_array / prob_array.sum()
			results.append(prob_array)

	return np.array(results)


# 自定義分詞函數
def chinese_tokenizer(text):
	# 使用 jieba 進行分詞
	words = jieba.cut(text, cut_all=False)

	# 定義要忽略的標點符號，可以擴展為包含中文標點
	punctuation = string.punctuation + "，。！？；：“”‘’（）《》、"

	# 過濾掉標點符號
	filtered_words = [word for word in words if word not in punctuation]

	return filtered_words


# 創建解釋器
explainer = LimeTextExplainer(
	# class_names=["症状诊断相关", "症状诊断不相关"],
	class_names=["中医病症诊断结果相关", "中医病症诊断结果不相关"],
	split_expression=chinese_tokenizer,  # 使用jieba分詞
	bow=True,
	random_state=42,
)

# 自動換行函數
def wrap_text(text, width=70):
	# 使用textwrap.fill()直接返回換行後的文本字符串
	wrapped_text = textwrap.fill(text, width=width)
	return wrapped_text

# 更多自定義選項的用法
def wrap_text_advanced(text, width=70, initial_indent="", subsequent_indent="  "):
	wrapper = textwrap.TextWrapper(
		width=width,  # 每行最大寬度
		initial_indent=initial_indent,  # 第一行縮進
		subsequent_indent=subsequent_indent,  # 後續行縮進
		break_long_words=True,  # 允許斷開長單詞
		break_on_hyphens=True,  # 允許在連字符處斷行
	)
	wrapped_text = wrapper.fill(text)
	return wrapped_text

# ************************ 模型部分代碼 結束 ************************



# ************************ Server API 部分代碼 ************************
# 啟動服務器 uvicorn api:app --host 0.0.0.0 --port 8000 --reload
from fastapi import FastAPI, HTTPException

# 創建 FastAPI 應用實例
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/query")
async def query_model(question):
    try:
        # 調用模型推理邏輯
        # "张某，男，27岁。患者因昨晚饮酒发热，喝凉水数杯，早晨腹痛腹泻，大便如水色黄，腹中辘辘有声，恶心欲吐，胸中满闷不舒，口干欲冷饮，舌质红、苔白腻，脉沉细数。给出中医诊断和处方建议。"
        response = get_model_response(question)
        print(response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fullQuery")
async def full_query_model(question):
    try:
        response = get_model_response(question)
        print(response)
        output = []
        output.append({"o_question": question, "o_response": response})
        print("使用LIME計算詞語貢獻度中...")

        # 生成解釋
        explanation = explainer.explain_instance(
            text_instance=question,
            classifier_fn=predictor,
            num_features=len(chinese_tokenizer(question)),
            num_samples=1000,
            labels=[0, 1],
        )

        # 自動提問
        for word_weight in explanation.as_list():
            word, weight = word_weight
            if weight>0:
                new_q = "词语\""+word+"\"如何影响了你刚才首次中医诊断的回答？"
                print("\n=== 病人提問 ===")
                print(wrap_text((new_q), width=30))

                print("\n=== 模型結果 ===")
                new_response = get_model_response(new_q)
                print(wrap_text(new_response, width=30))
            else:
                new_q = "词语\""+word+"\"为何对你刚才首次中医诊断的回答影响较小？我应该怎么样提问？"
                print("\n=== 病人提問 ===")
                print(wrap_text((new_q), width=30))

                print("\n=== 模型結果 ===")
                new_response = get_model_response(new_q)
                print(wrap_text(new_response, width=30))
            output.append({"question": new_q, "response": new_response})

        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("啟動API服務器中...")
    # 記得允許防火墻開放端口：sudo ufw allow 8000
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

