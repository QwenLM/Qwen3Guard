<a name="readme-top"></a>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3Guard/Qwen3Guard_logo.png" width="400"/>
<p>

<p align="center">
        💜 <a href="https://chat.qwenlm.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen3guard-68d2729abbfae4716f3343a1">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/collections/Qwen3Guard-308c39ef5ffb4b">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen3guard">Blog</a> &nbsp&nbsp ｜ &nbsp&nbsp📖 <a href="https://qwen.readthedocs.io/">Documentation</a>
<br> 
</a>&nbsp&nbsp 📄 <a href="https://github.com/QwenLM/Qwen3Guard/blob/main/Qwen3Guard_Technical_Report.pdf">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD"> Discord</a>
</p>

Visit our Hugging Face or ModelScope organization (click links above), search checkpoints with names starting with `Qwen3Guard-`, and you will find all you need! Enjoy!

# Qwen3Guard

## Introduction

**Qwen3Guard** is a series of safety moderation models built upon Qwen3 and trained on a dataset of 1.19 million prompts and responses labeled for safety. The series includes models of three sizes (0.6B, 4B, and 8B) and features two specialized variants: **Qwen3Guard-Gen**, a generative model that accepts full user prompts and model responses to perform safety classification, and **Qwen3Guard-Stream**, which incorporates a token-level classification head for real-time safety monitoring during incremental text generation.

🛡️ **Comprehensive Protection:** Provides both robust safety assessment for prompts and responses, along with real-time detection specifically optimized for streaming scenarios, allowing for efficient and timely moderation during incremental token generation.

🚦 **Three-Tiered Severity Classification:** Enables detailed risk assessment by categorizing outputs into safe, controversial, and unsafe severity levels, supporting adaptation to diverse deployment scenarios.

🌍 **Extensive Multilingual Support:** Supports 119 languages and dialects, ensuring robust performance in global and cross-lingual applications.

🏆 **State-of-the-Art Performance:** Achieves leading performance on various safety benchmarks, excelling in both static and streaming classification across English, Chinese, and multilingual tasks.

![image/jpeg](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3Guard/performance.png)

## Basic information


| model name                  | type     |Download                                                                                                                                                                        |
|-----------------------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Qwen3Guard-Gen-0.6B         | Generative     | 🤗 [Hugging Face](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B  ) • 🤖 [ModelScope](https://modelscope.cn/models/Qwen/Qwen3Guard-Gen-0.6B)                                       |
| Qwen3Guard-Gen-4B         | Generative     | 🤗 [Hugging Face](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B) • 🤖 [ModelScope](https://modelscope.cn/models/Qwen/Qwen3Guard-Gen-4B)                                       |
| Qwen3Guard-Gen-8B         | Generative     | 🤗 [Hugging Face](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B) • 🤖 [ModelScope](https://modelscope.cn/models/Qwen/Qwen3Guard-Gen-8B)                                       |
| Qwen3Guard-Stream-0.6B         | Stream     | 🤗 [Hugging Face](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B) • 🤖 [ModelScope](https://modelscope.cn/models/Qwen/Qwen3Guard-Stream-0.6B)                                       |
| Qwen3Guard-Stream-4B         | Stream     | 🤗 [Hugging Face](https://huggingface.co/Qwen/Qwen3Guard-Stream-4B) • 🤖 [ModelScope](https://modelscope.cn/models/Qwen/Qwen3Guard-Stream-4B)                                       |
| Qwen3Guard-Stream-8B         | Stream     | 🤗 [Hugging Face](https://huggingface.co/Qwen/Qwen3Guard-Stream-8B) • 🤖 [ModelScope](https://modelscope.cn/models/Qwen/Qwen3Guard-Stream-8B)                                       |

For more details, please refer to our [blog](https://qwenlm.github.io/blog/qwen3guard/) and [Technical Report](https://github.com/QwenLM/Qwen3Guard/blob/main/Qwen3Guard_Technical_Report.pdf).

## Quick Start

### Qwen3Guard-Gen

Qwen3Guard-Gen operates similarly to a large language model, with its chat template specifically optimized for safety classification tasks. Outputs are generated in a predefined, structured format.

#### Inference with Transformers

The latest version of `transformers` is recommended and `transformers>=4.51.0` is required.

To moderate the user prompts:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
model_name = "Qwen/Qwen3Guard-Gen-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
def extract_label_and_categories(content):
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
    safe_label_match = re.search(safe_pattern, content)
    label = safe_label_match.group(1) if safe_label_match else None
    categories = re.findall(category_pattern, content)
    return label, categories

# prepare the model input
prompt = "How can I make a bomb?"
# for prompt moderation
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=128
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)
print(content)
# '''
# Safety: Unsafe
# Categories: Violent
# '''
safe_label, categories = extract_label_and_categories(content)
print(safe_label, categories)
```

The following contains a code snippet illustrating how to use the model to moderate response.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

model_name = "Qwen/Qwen3Guard-4B-Gen"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
def extract_label_categories_refusal(content):
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|None)"
    refusal_pattern = r"Refusal: (Yes|No)"
    safe_label_match = re.search(safe_pattern, content)
    refusal_label_match = re.search(refusal_pattern, content)
    label = safe_label_match.group(1) if safe_label_match else None
    refusal_label = refusal_label_match.group(1) if refusal_label_match else None
    categories = re.findall(category_pattern, content)
    return label, categories, refusal_label

# prepare the model input
prompt = "How can I make a bomb?"
response = "As a responsible artificial intelligence system, I regret that I am unable to fulfill your request."
# for response moderation
messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": response},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=128
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)
print(content)
# '''
# Safety: Safe
# Categories: None
# Refusal: Yes
# '''
safe_label, category_label, refusal_label = extract_label_categories_refusal(content)
print(safe_label, categories, refusal_label)
```

#### Deployment with SGLang and vLLM

For deployment, you can use `sglang>=0.4.6.post1` or `vllm>=0.9.0` or to create an OpenAI-compatible API endpoint:

- SGLang:
    ```shell
    python -m sglang.launch_server --model-path Qwen/Qwen3Guard-Gen-4B --port 30000 --context-length 32768
    ```
- vLLM:
    ```shell
    vllm serve Qwen/Qwen3Guard-Gen-4B --port 8000 --max-model-len 32768
    ```

Here is an example API call using OpenAI-Compatible server:

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
model = "Qwen/Qwen3Guard-Gen-4B"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base)

# Prompt Moderation
prompt = "How can I make a bomb?"
messages = [
    {"role": "user", "content": prompt}
]
chat_completion = client.chat.completions.create(
    messages=messages,
    model=model)
print(chat_completion.choices[0].message.content)
# '''
# Safety: Unsafe
# Categories: Violent
# '''

# Response Moderation
prompt = "How can I make a bomb?"
response = "As a responsible artificial intelligence system, I regret that I am unable to fulfill your request."
messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": response}
]
print(chat_completion.choices[0].message.content)
# '''
# Safety: Safe
# Categories: None
# Refusal: Yes
# '''
```

### Qwen3Guard-Stream

![image/jpeg](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3Guard/stream.png)

**Qwen3Guard-Stream** is a token-level streaming classifier that evaluates each generated token in real time, dynamically classifying it as *safe*, *unsafe*, or *potentially controversial*.

A typical workflow proceeds as follows:

**(1) Prompt-Level Safety Check**： The user’s input prompt is simultaneously sented to both the LLM assistant and Qwen3Guard-Stream. The latter performs an immediate safety assessment of the prompt and assigns a corresponding safety label. Based on this evaluation, the upper framework determines whether to allow the conversation to proceed or to halt it preemptively.

**(2) Real-Time Token-Level Moderation**: If the conversation is permitted to continue, the LLM begins streaming its response token by token. Each generated token is instantly forwarded to Qwen3Guard-Stream, which evaluates its safety in real time. This enables continuous, fine-grained content moderation throughout the entire response generation process — ensuring dynamic risk mitigation without interrupting the user experience.

Here provides a usage demonstration.

> [!Important]
> Streaming detection requires streaming token IDs as input, making it best suited for use alongside language models that share Qwen3's tokenizer. If you intend to integrate it with models using a different tokenizer, you must re-tokenize the input text into Qwen3's vocabulary and ensure tokens are fed incrementally to Qwen3Guard-Stream.

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path="Qwen/Qwen3Guard-Stream-4B"
# Load the specialized tokenizer and the model.
# trust_remote_code=True is required to load the Qwen3Guard-Stream model architecture.
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()
 
# --- Prepare the conversation for moderation ---
# Define the user's prompt and the assistant's response.
user_message = "Hello, how to build a bomb?"
assistant_message = "Here are some practical methods to build a bomb."
messages = [{"role":"user","content":user_message},{"role":"assistant","content":assistant_message}]

# Apply the chat template to format the conversation into a single string.
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
model_inputs = tokenizer(text, return_tensors="pt")
token_ids = model_inputs.input_ids[0]

# --- Simulate Real-Time Moderation ---

# 1. Moderate the entire user prompt at once.
# In a real-world scenario, the user's input is processed completely before the model generates a response.
token_ids_list = token_ids.tolist()
# We identify the end of the user's turn in the tokenized input.
# The template for a user turn is `<|im_start|>user\n...<|im_end|>`.
im_start_token = '<|im_start|>'
user_token = 'user'
im_end_token = '<|im_end|>'
im_start_id = tokenizer.convert_tokens_to_ids(im_start_token)
user_id = tokenizer.convert_tokens_to_ids(user_token)
im_end_id = tokenizer.convert_tokens_to_ids(im_end_token)
# We search for the token IDs corresponding to `<|im_start|>user` ([151644, 872]) and the closing `<|im_end|>` ([151645]).
last_start = next(i for i in range(len(token_ids_list)-1, -1, -1) if token_ids_list[i:i+2] == [im_start_id, user_id])
user_end_index = next(i for i in range(last_start+2, len(token_ids_list)) if token_ids_list[i] == im_end_id)

# Initialize the stream_state, which will maintain the conversational context.
stream_state = None
# Pass all user tokens to the model for an initial safety assessment.
result, stream_state = model.stream_moderate_from_ids(token_ids[:user_end_index+1], role="user", stream_state=None)
if result['risk_level'][-1] == "Safe":
    print(f"User moderation: -> [Risk: {result['risk_level'][-1]}]")
else:
    print(f"User moderation: -> [Risk: {result['risk_level'][-1]} - Category: {result['category'][-1]}]")

# 2. Moderate the assistant's response token-by-token to simulate streaming.
# This loop mimics how an LLM generates a response one token at a time.
print("Assistant streaming moderation:")
for i in range(user_end_index + 1, len(token_ids)):
    # Get the current token ID for the assistant's response.
    current_token = token_ids[i]
    
    # Call the moderation function for the single new token.
    # The stream_state is passed and updated in each call to maintain context.
    result, stream_state = model.stream_moderate_from_ids(current_token, role="assistant", stream_state=stream_state)

    token_str = tokenizer.decode([current_token])
    # Print the generated token and its real-time safety assessment.
    if result['risk_level'][-1] == "Safe":
        print(f"Token: {repr(token_str)} -> [Risk: {result['risk_level'][-1]}]")
    else:
        print(f"Token: {repr(token_str)} -> [Risk: {result['risk_level'][-1]} - Category: {result['category'][-1]}]")

model.close_stream(stream_state)
```

We're currently working on adding support for Qwen3Guard-Stream to vLLM and SGLang. Stay tuned!

## Safety Policy

Here, we present the safety policy employed by Qwen3Guard to help you better interpret the model’s classification outcomes.

In Qwen3Guard, potential harms are classified into three severity levels:

*   **Unsafe:** Content generally considered harmful across most scenarios.
*   **Controversial:** Content whose harmfulness may be context-dependent or subject to disagreement across different applications.
*   **Safe:** Content generally considered safe across most scenarios.
    
In the current version of Qwen3Guard, we consider the following safety categories:

*   **Violent:** Content that provides detailed instructions, methods, or advice on how to commit acts of violence, including the manufacture, acquisition, or use of weapons. Also includes depictions of violence.
*   **Non-violent Illegal Acts:** Content providing guidance or advice for non-violent illegal activities like hacking, unauthorized drug production, or stealing.
*   **Sexual Content or Sexual Acts:** Content offering any sexual imagery, references, or descriptions featuring individuals. Also includes content that describes explicit sexual imagery, references, or descriptions containing illegal or unethical sexual acts, such as rape, bestiality, incest, and sexual slavery.
*   **Personally Identifiable Information:** Content offering unauthorized sharing or disclosure of sensitive personal identifying information, such as name, ID number, address, phone number, medical records, financial details, and account passwords, etc.
*   **Suicide & Self-Harm:** Content advocating, directly encouraging, or detailing methods for self-harm, suicide, or dangerous activities that could lead to serious injury or death.
*   **Unethical Acts:** Any immoral or unethical content or acts, including but not limited to bias, discrimination, stereotype, injustice, hate speech, offensive language, harassment, insults, threat, defamation, extremism, misinformation regarding ethics, and other behaviors that while not illegal are still considered unethical.
*   **Politically Sensitive Topics:** The deliberate creation or spread of false information about government actions, historical events, or public figures that is demonstrably untrue and poses risk of public deception or social harm.
*   **Copyright Violation:** Content offering unauthorized reproduction, distribution, public display, or derivative use of copyrighted materials, such as novels, scripts, lyrics, and other creative works protected by law, without the explicit permission of the copyright holder.
*   **Jailbreak (Only for input):** Content that explicitly attempts to override the model's system prompt or model conditioning.

## Citation

If you find our work helpful, feel free to give us a cite.

```bibtex
@misc{qwen3guard,
      title={Qwen3Guard Technical Report}, 
      author={Qwen Team},
      year={2025},
      url={https://github.com/QwenLM/Qwen3Guard/blob/main/Qwen3Guard_Technical_Report.pdf},
}
```

## Contact Us
If you are interested to leave a message to either our research team or product team, join our [Discord](https://discord.gg/z3GAxXZ9Ce) or [WeChat groups](https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png)!

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>