"""
Train and finetune the colbert-ir/colbertv2 model with the uber 10k dataset.
LlamaIndex has provided https://llamahub.ai/?tab=llama_datasets, this file can train and fine tune
the ColBERT model with these datasets. 

I ran this file on CUDA for better performance.

download dataset from llama-index
$ llamaindex-cli download-llamadataset Uber10KDataset2021 --download-dir ./data/uber10k
"""
from multiprocessing import freeze_support
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_dataset import LabelledRagDataset

print("loading...")
rag_dataset = LabelledRagDataset.from_json("./data/uber10k/rag_dataset.json")
pairs = [(example.query, example.reference_answer) for example in rag_dataset.examples]
print(f"uber10k rag dataset size {len(pairs)}")
documents = SimpleDirectoryReader(input_dir="./data/uber10k/source_files").load_data()

corpus = [doc.text for doc in documents]

print(f"entire doc size {len(corpus)}")

from ragatouille import RAGTrainer

if __name__ == '__main__':
    freeze_support()
    trainer = RAGTrainer(model_name = "MyFineTunedUber10kColBERT",
        pretrained_model_name = "colbert-ir/colbertv2.0")

    trainer.prepare_training_data(raw_data=pairs,
                                data_out_path="./data/",
                                all_documents=corpus)

    trainer.train(batch_size=32)
