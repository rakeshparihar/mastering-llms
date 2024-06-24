# Mastering LLMs: A Comprehensive Tutorial and Use Cases

Welcome to the **Mastering LLMs** repository! This repository contains a series of Jupyter notebooks designed to guide you through building various applications using Large Language Models (LLMs). Each notebook covers a different aspect of LLM development, from basic applications to advanced techniques.

## Notebooks Overview

### 1. Create a Basic LLM Application
- **Objective:** Create a simple application using OpenAI's GPT-4 model to generate text from a prompt.
- **Notebook:** [Create a Basic LLM Application](./1._Create_a_Basic_LLM_Application.ipynb)

### 2. Chain Prompts for Complex Tasks
- **Objective:** Learn how to connect multiple prompts for advanced objectives.
- **Notebook:** [Chain Prompts for Complex Tasks](./2._Chain_Prompts_for_Complex_Tasks.ipynb)

### 3. Implement RAG (Retrieval-Augmented Generation)
- **Objective:** Integrate external knowledge bases to enhance your models using Retrieval-Augmented Generation.
- **Notebook:** [Implement RAG (Retrieval-Augmented Generation)](./3._Implement_RAG_(Retrieval-Augmented_Generation).ipynb)

### 4. Add Memory for Contextual Understanding
- **Objective:** Enable memory in your models for better contextual understanding.
- **Notebook:** [Add Memory for Contextual Understanding](./4._Add_Memory_for_Contextual_Understanding.ipynb)

### 5. Interact with External Tools
- **Objective:** Teach your LLMs to use APIs and other external resources.
- **Notebook:** [Interact with External Tools](./5._Interact_with_External_Tools.ipynb)

### 6. Develop LLM Agents
- **Objective:** Build agents capable of autonomous decision-making.
- **Notebook:** [Develop LLM Agents](./6._Develop_LLM_Agents.ipynb)

### 7. Fine-Tune with PEFT Methods
- **Objective:** Explore advanced techniques to fine-tune your models for specific tasks.
- **Notebook:** [Fine-Tune with PEFT Methods](./7._Fine-Tune_with_PEFT_Methods.ipynb)

## Getting Started

### Prerequisites

To run the notebooks, you need to have the following installed:
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/rakeshparihar/llm-usecases.git
    cd llm-usecases
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Notebooks

1. Start Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```

2. Open the notebook you want to explore and follow the instructions inside.

## Detailed Instructions

### 1. Create a Basic LLM Application
- **Objective:** Create a simple application using OpenAI's GPT-4 model to generate text from a prompt.
- **Code Overview:**
    ```python
    import openai

    # Initialize OpenAI API
    openai.api_key = 'your-api-key'

    def generate_text(prompt):
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()

    # Example usage
    prompt = "Once upon a time in a land far away,"
    result = generate_text(prompt)
    print(result)
    ```

### 2. Chain Prompts for Complex Tasks
- **Objective:** Learn how to connect multiple prompts for advanced objectives.
- **Code Overview:**
    ```python
    import openai

    openai.api_key = 'your-api-key'

    def generate_text(prompt):
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()

    prompt1 = "The scientist discovered a new element. It had unusual properties, such as"
    generated_text1 = generate_text(prompt1)

    prompt2 = generated_text1 + " This discovery led to"
    result = generate_text(prompt2)
    print(result)
    ```

### 3. Implement RAG (Retrieval-Augmented Generation)
- **Objective:** Integrate external knowledge bases to enhance your models using Retrieval-Augmented Generation.
- **Code Overview:**
    ```python
    import openai

    openai.api_key = 'your-api-key'

    def generate_text(prompt):
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()

    question = "What are the benefits of quantum computing?"
    result = generate_text(question)
    print(result)
    ```

### 4. Add Memory for Contextual Understanding
- **Objective:** Enable memory in your models for better contextual understanding.
- **Code Overview:**
    ```python
    import openai

    openai.api_key = 'your-api-key'
    conversation_history = []

    def generate_text_with_memory(prompt):
        conversation_history.append(prompt)
        combined_prompt = "\\n".join(conversation_history)
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=combined_prompt,
            max_tokens=100
        )
        result = response.choices[0].text.strip()
        conversation_history.append(result)
        return result

    prompt = "Hello, how are you?"
    response = generate_text_with_memory(prompt)
    print(response)
    ```

### 5. Interact with External Tools
- **Objective:** Teach your LLMs to use APIs and other external resources.
- **Code Overview:**
    ```python
    import openai
    import requests

    openai.api_key = 'your-api-key'

    def generate_text(prompt):
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()

    prompt = "Generate a request to fetch current weather data for New York City."
    api_call = generate_text(prompt)
    print(api_call)

    response = requests.get("http://api.weatherapi.com/v1/current.json?key=YOUR_API_KEY&q=New York City")
    weather_data = response.json()
    print(weather_data)
    ```

### 6. Develop LLM Agents
- **Objective:** Build agents capable of autonomous decision-making.
- **Code Overview:**
    ```python
    import openai

    openai.api_key = 'your-api-key'

    def decision_agent(prompt):
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()

    prompt = "You are a financial advisor. A client asks whether they should invest in stocks or real estate. What do you advise?"
    decision = decision_agent(prompt)
    print(decision)
    ```

### 7. Fine-Tune with PEFT Methods
- **Objective:** Explore advanced techniques to fine-tune your models for specific tasks.
- **Code Overview:**
    ```python
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    model = GPT2LMHeadModel.from_pretrained("gpt-2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt-2")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    trainer.train()
    ```

## Contributing

Feel free to fork this repository, make improvements, and submit a pull request. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
