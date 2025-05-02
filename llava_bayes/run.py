import torch
import requests
from PIL import Image
from transformers import AutoProcessor, LlavaConfig # Assuming LlavaConfig is needed by your custom model
import gc
import numpy as np
from sentence_transformers import SentenceTransformer


model_id = "llava-hf/llava-1.5-7b-hf" # Use the specific LLaVa version you want
device = "cuda"
print("Loading processor...")
try:
    processor = AutoProcessor.from_pretrained(model_id)
    print("Processor loaded successfully.")
except Exception as e:
    print(f"Error loading processor: {e}")
    exit()


from my_llava import CustomLlavaModel

print("Loading the model...")
model = CustomLlavaModel.from_pretrained(
    model_id,
    torch_dtype = torch.bfloat16, 
    device_map = "auto"
)

print(f"Model device: {model.device}")

device = model.device

# prompt = "USER: <image>\n What game is the child playing?\n ASSISTANT:"
prompt = "USER: <image>\n What is color of the jersey which the child is wearing?\n ASSISTANT:"
# prompt = "USER: <image>\n What is the total count of remotes and cats in the image combined?\n ASSISTANT:"
# prompt = "USER: <image>\n How many balls are in the image?\n ASSISTANT:"


# image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
# image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_url = "https://cdn.pixabay.com/photo/2015/01/26/22/40/child-613199_1280.jpg"
# image_url = "https://tennex.in/cdn/shop/files/Cricket_Tennis_Ball_Premier_Light.png?v=1728326779"
raw_image = Image.open(requests.get(image_url, stream=True).raw)
print("Image downloaded.")
# Process inputs, ensuring they are tensors ready for the model
inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
# def decode(ids):
#     output_text = processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#     return output_text

print("Generating answer from the model")
answers = []
K = 16
with torch.no_grad():
    for i in range(K):
        generate_ids = model.generate(**inputs, max_new_tokens=64)
        x = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        answers.append(x)


question = answers[0].split("ASSISTANT: ",1)[0]
print("Question: ")
print(question)
print("#############################")
print("Sampled Answers: ")
for i,a in enumerate(answers):
    print(f"[{i}]: {a.split('ASSISTANT: ', 1)[1]}")
    # print("")


def get_embedding_distribution(texts: list[str], model_name: str = 'all-MiniLM-L6-v2'):
    if not texts:
        print("Warning: No texts provided.")
        return None, None, None

    K = len(texts)
    print(f"Processing {K} texts...")

    print(f"Loading sentence transformer model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Model loaded.")

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Embeddings generated. Shape: {embeddings.shape}") # (K, embedding_dim)

    if K == 0: # Should have been caught earlier, but for safety
         return None, None, embeddings # embeddings will be empty array
    elif K == 1:
        print("Only one text provided. Covariance matrix will be all zeros.")
        mean_vector = embeddings[0]
        embedding_dim = embeddings.shape[1]
        covariance_matrix = np.zeros((embedding_dim, embedding_dim))
    else:
        print("Calculating mean vector...")
        mean_vector = np.mean(embeddings, axis=0)

        print("Calculating covariance matrix...")
        covariance_matrix = np.cov(embeddings, rowvar=False)

    print(f"Mean vector shape: {mean_vector.shape}")
    print(f"Covariance matrix shape: {covariance_matrix.shape}")

    return mean_vector, covariance_matrix, embeddings

list_of_texts = answers

K = len(list_of_texts)

if K > 0:
    mean_vec, cov_matrix, all_embeddings = get_embedding_distribution(list_of_texts)

    if mean_vec is not None:
        print("\n--- Results ---")
        print(f"Calculated for {K} texts.")
        print(f"\nMean Vector (first 5 dimensions):\n{mean_vec[:5]}")
        print(f"\nCovariance Matrix (top-left 5x5 submatrix):\n{cov_matrix[:5, :5]}")

        trace = np.trace(cov_matrix)
        print(f"\nTrace of Covariance Matrix (Sum of Variances): {trace:.4f}")

        eigenvalues = np.linalg.eigvalsh(cov_matrix) # Use eigvalsh for symmetric matrices
        print(f"\nEigenvalues (showing largest 5):\n{np.sort(eigenvalues)[::-1][:5]}")
        print(f"Sum of eigenvalues (should approx equal trace): {np.sum(eigenvalues):.4f}")

else:
    print("Please provide at least one text in the list.")
