# Prompt Engineering

## 🚀 Overview

Prompt Engineering is the art of designing inputs to optimize AI responses. It helps guide AI to provide more accurate, useful, and structured outputs.

This module explores four key techniques:
1. **Zero-Shot Prompting** - Asking AI a question without examples.
2. **Few-Shot Prompting** - Providing a few examples to guide AI.
3. **Chain-of-Thought (CoT) Prompting** - Encouraging AI to reason step by step.
4. **Self-Consistency CoT Prompting** - Running multiple responses and selecting the most common one.

---

## 🧠 Understanding These Concepts

### 🎓 For a 10-Year-Old
1. **Zero-Shot Prompting** - Imagine asking a friend a question without giving any clues, and they still figure it out!
2. **Few-Shot Prompting** - Showing your friend a few examples first so they understand what kind of answer you expect.
3. **Chain-of-Thought Prompting** - Asking your friend to explain their thinking step by step before giving the final answer.
4. **Self-Consistency CoT Prompting** - Asking multiple friends the same question and choosing the answer most of them agree on.

### 🎓 For a 25-Year-Old
1. **Zero-Shot Prompting** - The AI is expected to generate a response without prior context or examples.
2. **Few-Shot Prompting** - AI is provided with a few examples to infer the desired pattern.
3. **Chain-of-Thought Prompting** - AI is encouraged to break down reasoning into logical steps to improve accuracy.
4. **Self-Consistency CoT Prompting** - Running multiple independent CoT reasoning processes and selecting the most consistent output.

---

## Demo
<a href="https://huggingface.co/spaces/Ganesh-Kunnamkumarath/Prompt_Engineering" target="_blank">
Demo on Hugging face spaces
</a>


## 📂 Folder Structure

```
/prompt_engineering/
│── code/
│   │── 1.zero_shot_prompting.py
│   │── 2.few_shot_prompting.py
│   │── 3.chain-of-thought(CoT)-prompting.py
│   │── 4.self-consistency-CoT-prompting.py
│── README.md
│── references.md
│── requirements.txt
```

---

## 🛠 Running the Code

### 1️⃣ Install Dependencies
Make sure you have Python installed. Then, install dependencies:
```bash
pip install -r requirements.txt
```

### 2️⃣ Set Up OpenAI API Key
Store your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

### 3️⃣ Run Each Script
Execute any of the scripts:
```bash
python code/1.zero_shot_prompting.py
python code/2.few_shot_prompting.py
python code/3.chain-of-thought(CoT)-prompting.py
python code/4.self-consistency-CoT-prompting.py
```

---

## 📚 References
For more learning resources, check the `references.md` file for official documentation and tutorials.
