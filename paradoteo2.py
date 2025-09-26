import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

#sentnences we exmine
original_sentences = ["Hope you too, to enjoy it as my deepest wishes", "Because I didnt see that part final yet, or maybe I missed, I apologize if so"]

reconstructed_sentences = ["I hope you enjoy it with my deepest wishes too", "Because I didn't see that part finished yet, or maybe I missed, I apologize if so"]

#word embeddings with a BERT model

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  #no training, inference mode


def get_bert_embedding(sentence):
    
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    #mean pooling all token embeddings
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

#get the embedings of all sentences
emb_original = [get_bert_embedding(s) for s in original_sentences]
emb_reconstructed = [get_bert_embedding(s) for s in reconstructed_sentences]

#cosine simularity calculation for comparison
print("\nCosine Similarity Scores:")
for i, (orig, recon) in enumerate(zip(emb_original, emb_reconstructed)):
    sim = cosine_similarity([orig], [recon])[0][0]
    print(f"Pair {i+1}: {sim:.4f}")


all_sentences = original_sentences + reconstructed_sentences
all_embeddings = np.vstack(emb_original + emb_reconstructed)

#PCA for 2D matrix
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_embeddings)

#t-SNE for a more "semantic" visualization
tsne = TSNE(n_components=2, perplexity=2, random_state=42)
tsne_result = tsne.fit_transform(all_embeddings)


#visualization
def plot_embeddings(points, title, labels):
    plt.figure(figsize=(8, 6))
    colors = ['red', 'red', 'blue', 'blue']  # red = original, blue = reconstructed
    for i, (x, y) in enumerate(points):
        plt.scatter(x, y, color=colors[i])
        plt.text(x + 0.01, y + 0.01, labels[i], fontsize=9)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.show()


labels = ["Original 1", "Original 2","Reconstructed 1", "Reconstructed 2"]
plot_embeddings(pca_result, "PCA Visualization of Sentence Embeddings", labels)

plot_embeddings(tsne_result, "t-SNE Visualization of Sentence Embeddings", labels)
