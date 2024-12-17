# Code Detective Clone Detection

Detect semantically similar python code using fine-tuned GraphCodeBERT model.

## About

This modified [GraphCodeBERT](https://arxiv.org/abs/2009.08366) model was fine-tuned for 11 hours using an A40 server on the [C4 dataset](https://github.com/Chenning-Tao/C4/tree/main) dataset, which contains semantically similar python code snippets.

It is then used to predict the similarity of python code snippets. It achieved F1 scores of greater than 0.96 on all datasets in several experiments, where balanced sampling was applied.

## Prerequisites & Installation

* Activate Virual Env

```python3 -m venv myenv ```
source myenv/bin/activate
206819985@OTS-FVFHK1DWQ05Q-MAC PythonCodeDetective % pip install -r requirements.txt ```


    In your virtual environment, run:

    ```sh
    pip install -r requirements.txt
    ```

    to install the required packages.

    NOTE: In your local system, you need to install this as well : pip install 'transformers[torch]' accelerate


* conda

    To create a new conda environment called `PythonCloneDetection` with the required packages, run:

    ```sh
    conda env create -f environment.yml
    ```

    (this may take a while to finish)

The above commands will install cpu-only version of the `pytorch` package. Please refer to [PyTorch's official website](https://pytorch.org/get-started/locally/) for instructions on how to install other versions of `pytorch` on your machine.

## Usage

1. Run `python main.py --input <input_path> --output <output_path>` to run `CloneClassifier` on the csv file at `<input_path>` and save its predictions at `<output_path>`. For example:

    ```sh
    python main.py --input examples/c4.csv --output results/res.csv
    ```

    The input of `main.py` is a csv file containing two columns named `code1` and `code2`, where each row contains a pair of python code snippets to be compared. The output csv file will have three columns named `code1`, `code2`, and `predictions`, where `predictions` indicates whether the two code snippets in the corresponding row are semantically similar.

2. Use the command `python main.py --help` to see other optional arguments including `max_token_size`, `fp16`, and `per_device_eval_batch_size`.
3. You could also import `CloneClassifier` class from `clone_classifier.py` and use it in your own code, for example:

    ```python
    import pandas as pd
    from clone_classifier import CloneClassifier


    classifier = CloneClassifier(
        max_token_size=512,
        fp16=False,  # set to True for faster inference if available
        per_device_eval_batch_size=8,
    )

    df = pd.read_csv("examples/c4.csv").head(10)
    res_df = classifier.predict(
        df[["code1", "code2"]], 
        # save_path="results/res.csv"
    )

    print(res_df["predictions"] == df["similar"])
    ```

#Tools & DataSet

## Dataset Creation

### Dataset 1: C4

#### Step 1: Clone the C4 Dataset
1. Clone the C4 repository:
   ```bash
   git clone https://github.com/Chenning-Tao/C4.git

#### 2. Install & Configure lfs

git lfs install

git lfs pull

#### 3: Navigate to the cloned directory and pull LFS files:
cd C4

git lfs pull

### Step 2: Convert JSONL to CSV
The dataset files are in .jsonl format and contain data in multiple languages. To format the dataset and extract only Python-related data, run the provided script for the train, test, and validation datasets:

### Step3: Real time testing
``` bash python detective_main.py --input examples/c6.csv --output results/res6.csv```
### Step4: Evaluation
```bash python evaluate/eval.py c4_dataset/C4/dataset/test.csv results/test_result.csv```

