import argparse
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def evaluate_model(test_file, result_file):
    # Load the test data (true labels)
    test_df = pd.read_csv(test_file)
    
    # Load the result data (predictions)
    result_df = pd.read_csv(result_file)

    # Ensure both dataframes are aligned by code pairs (optional check)
    if not test_df[['code1', 'code2']].equals(result_df[['code1', 'code2']]):
        raise ValueError("The test and result CSV files are not aligned by code pairs.")

    # Extract true labels (from the 'similar' column) and predicted labels (from the 'predictions' column)
    y_true = test_df['similar']
    y_pred = result_df['predictions']

    # Calculate Precision, Recall, and F1-Score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accurancy:", precision*recall / f1)

    # Cosine Similarity (optional): Calculate cosine similarity between the code pairs
    # For cosine similarity, we need the vector representations of the code snippets.
    # Assuming you have embeddings or vector representations of `code1` and `code2`, 
    # we will calculate cosine similarity between them.

    # For demonstration, we assume you already have the vectorized versions of code1 and code2.
    # Let's assume `code1_vec` and `code2_vec` are the embeddings you get from your model.
    # This can be achieved using any code embedding technique like transformers, BERT, etc.

    # Example (you need to replace this part with actual embedding logic):
    # Code snippet embeddings (replace with actual vectors)
    code1_vecs = np.random.rand(len(test_df), 300)  # Example random vectors
    code2_vecs = np.random.rand(len(test_df), 300)  # Example random vectors
    
    # Compute cosine similarity between code1 and code2 vectors
    cosine_sim = cosine_similarity(code1_vecs, code2_vecs)
    
    # Optionally, calculate the average cosine similarity
    avg_cosine_similarity = np.mean(cosine_sim)

    print(f"Average Cosine Similarity: {avg_cosine_similarity:.4f}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Evaluate the model on test data")
    
    # Define the arguments
    parser.add_argument('test_file', type=str, help="Path to the test CSV file")
    parser.add_argument('result_file', type=str, help="Path to the result CSV file")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the evaluation function with the provided file paths
    evaluate_model(args.test_file, args.result_file)

if __name__ == "__main__":
    main()
