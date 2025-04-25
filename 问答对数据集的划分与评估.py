# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: bge_base_zh_eval.py
# @time: 2024/6/6 16:32
import os
import json
import time
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainer

start_time = time.time()
project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]

# Function to load and process individual QA files
def load_individual_qa_files(directory):
    qa_pairs_with_source = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                source = content.get("source")
                qa_pairs = content.get("qa_pairs", [])
                for qa_pair in qa_pairs:
                    qa_pair["source"] = source
                    qa_pairs_with_source.append(qa_pair)
    return qa_pairs_with_source

# Load individual QA files
individual_qa_directory = os.path.join(project_dir, "data/individual_qa")
individual_qa_pairs = load_individual_qa_files(individual_qa_directory)

# Split data into training and evaluation sets
train_qa_pairs, eval_qa_pairs = train_test_split(individual_qa_pairs, test_size=0.2, random_state=42)

# Prepare training dataset
train_anchor, train_positive = [], []
for i, pair in enumerate(train_qa_pairs):
    train_anchor.append(pair["question"])
    train_positive.append(pair["answer"])

train_dataset = Dataset.from_dict({"positive": train_positive, "anchor": train_anchor})

print(train_dataset)
print(train_dataset[0:5])

# Prepare evaluation dataset
eval_corpus = {f"d{i}": {"text": pair["answer"]} for i, pair in enumerate(eval_qa_pairs)}
eval_queries = {f"q{i}": pair["question"] for i, pair in enumerate(eval_qa_pairs)}

# Create relevant_docs mapping for evaluation set
eval_relevant_docs = {}
for i in range(len(eval_qa_pairs)):
    q_id = f"q{i}"
    d_id = f"d{i}"
    eval_relevant_docs[q_id] = [d_id]

# Load a model
model_name = 'bge-base-zh-v1.5'
# 替换成自己的模型完整路径或使用huggingface model id
model_path = os.path.join(project_dir, f"models/{model_name}")
model = SentenceTransformer(model_path, device="cuda:0" if torch.cuda.is_available() else "cpu")
print("Model loaded")

# Evaluate the model before fine-tuning
evaluator_before_ft = InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=eval_corpus,
    relevant_docs=eval_relevant_docs,
    name=f"{model_name}_before_ft",
    score_functions={"cosine": cos_sim}
)

s_time_before_ft = time.time()
result_before_ft = evaluator_before_ft(model)
pprint(result_before_ft)
print(f"Evaluation time before fine-tuning: {time.time() - s_time_before_ft:.2f}s")

# Define loss function
train_loss = MultipleNegativesRankingLoss(model)

# Define training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=f"ft_{model_name}",  # output directory and hugging face model ID
    num_train_epochs=5,  # number of epochs
    per_device_train_batch_size=2,  # train batch size
    gradient_accumulation_steps=2,  # for a global batch size of 512
    per_device_eval_batch_size=4,  # evaluation batch size
    warmup_ratio=0.1,  # warmup ratio
    learning_rate=2e-5,  # learning rate, 2e-5 is a good value
    lr_scheduler_type="cosine",  # use constant learning rate scheduler
    optim="adamw_torch_fused",  # use fused adamw optimizer
    tf32=True,  # use tf32 precision
    bf16=True,  # use bf16 precision
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="epoch",  # evaluate after each epoch
    save_strategy="epoch",  # save after each epoch
    logging_steps=10,  # log every 10 steps
    save_total_limit=3,  # save only the last 3 models
    load_best_model_at_end=True,  # load the best model when training ends
    metric_for_best_model=f"eval_{model_name}_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score
)

# Train the model
trainer = SentenceTransformerTrainer(
    model=model,  # the model to train
    args=args,  # training arguments
    train_dataset=train_dataset.select_columns(["positive", "anchor"]),  # training dataset
    loss=train_loss,
    evaluator=evaluator_before_ft  # Use the same evaluator for consistency
)

trainer.train()

# Save the fine-tuned model
fine_tuned_model_path = os.path.join(project_dir, f"models/ft_{model_name}")
trainer.save_model(output_dir=fine_tuned_model_path)
print(f"Fine-tuned model saved to {fine_tuned_model_path}")

# Reload the fine-tuned model for evaluation
fine_tuned_model = SentenceTransformer(fine_tuned_model_path, device="cuda:0" if torch.cuda.is_available() else "cpu")
print("Fine-tuned model loaded")

# Evaluate the fine-tuned model
evaluator_after_ft = InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=eval_corpus,
    relevant_docs=eval_relevant_docs,
    name=f"{model_name}_after_ft",
    score_functions={"cosine": cos_sim}
)

s_time_after_ft = time.time()
result_after_ft = evaluator_after_ft(fine_tuned_model)
pprint(result_after_ft)
print(f"Evaluation time after fine-tuning: {time.time() - s_time_after_ft:.2f}s")

print(f"Total cost time: {time.time() - start_time:.2f}s")



