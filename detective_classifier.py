import tempfile
import multiprocessing

import torch
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments

from utils.encoder import Encoder
from utils.trainer import ImprovedRDropTrainer
from utils.collator import DataCollatorWithPadding
from utils.preprocessor import AnnotationPreprocessor, FunctionPreprocessor


class CloneClassifier:
    """
    A class that integrates data preprocessing, input tokenization, and model
    inferencing. It takes in a pandas dataframe with two columns: "code1" and
    "code2", and returns the predictions as a dataframe.
    """

    # Path to the locally saved model
    PLM_PATH = "./our_model"  # Path to your saved model directory

    def __init__(
        self,
        max_token_size=512,
        fp16=False,
        per_device_eval_batch_size=32,
    ):
        # -- Tokenizing & Encoding
        print("Load tokenizer and model from the saved directory")
        self.tokenizer = AutoTokenizer.from_pretrained(self.PLM_PATH)
        
        self.encoder = Encoder(self.tokenizer, max_input_length=max_token_size)

        print("Collator")
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, max_length=max_token_size
        )

        print("-- Config & Model")
        self.model = AutoModel.from_pretrained(self.PLM_PATH, trust_remote_code=True)

        with tempfile.TemporaryDirectory() as tmpdirname:
            training_args = TrainingArguments(
                output_dir=tmpdirname,  # output_dir is not needed for inference
                per_device_eval_batch_size=per_device_eval_batch_size,
                fp16=fp16,
                remove_unused_columns=False,
            )
            self.trainer = ImprovedRDropTrainer(
                model=self.model,
                args=training_args,
                data_collator=self.data_collator,
            )

    def prepare_inputs(self, df: pd.DataFrame):
        print("""Data preprocessing and tokenization.""")
        # -- Loading datasets
        dset = Dataset.from_pandas(df)

        print(" -- Preprocessing datasets")
        CPU_COUNT = multiprocessing.cpu_count() // 2
        print("at cpu_count", CPU_COUNT )
        fn_preprocessor = FunctionPreprocessor()
        dset = dset.map(fn_preprocessor, batched=True, num_proc=CPU_COUNT)

        an_preprocessor = AnnotationPreprocessor()
        dset = dset.map(an_preprocessor, batched=True, num_proc=CPU_COUNT)
        print("at about end of prepare_inputs", )
        dset = dset.map(
            self.encoder,
            batched=True,
            num_proc=multiprocessing.cpu_count(),
            remove_columns=dset.column_names,
        )

        return dset

    def predict(
        self, df: pd.DataFrame, save_path: str = None, return_score: bool = False
    ):
        """Perform model inference and return predictions as a dataframe."""
        print("at predict Preparing inputs")
        
        dset = self.prepare_inputs(df)

        
        print("-- Inference")
        outputs = self.trainer.predict(dset)[0]  # logits output
        print('logits outputs',outputs)
        scores = torch.Tensor(outputs).softmax(dim=-1).numpy()  # probability output

        results = df[["code1", "code2"]].copy()
        results["predictions"] = np.argmax(scores, axis=-1)
        if return_score:
            print("score of positive class")
            if scores.size == 1:
                results["score"] = scores
            else:
                results["score"] = scores[:, 1]

        if save_path is not None:
            results.to_csv(save_path, index=False)

        return results
