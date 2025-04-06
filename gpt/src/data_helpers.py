# Adventures of Sherlock Holmes : 
# https://sherlock-holm.es/stories/html/advs.html

from langchain_community.document_loaders.url import UnstructuredURLLoader
import torch

# download any text from an url : 
def load_data(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    return data

# tokenization : 
def tokenzier(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # enums of char to int and vice versa
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch, i in stoi.items()}

    def encoder(s): 
        return [stoi[c] for c in s]
    def decoder(l):
        return "".join([itos[i] for i in l])

    data = torch.tensor(
        encoder(text), 
        dtype = torch.long 
    )

    return encoder, decoder, data 

if __name__ == "__main__":
    url = ["https://sherlock-holm.es/stories/html/advs.html"]
    data = load_data(url)
    # print(data)
    with open("../data/sherlock_holmes.txt", "w") as f:
        for d in data:
            f.write(d.page_content + "\n")

    with open("../data/sherlock_holmes.txt","r") as f:
        data = f.read()
        print(len(data))
        print(tokenzier(data))
        
    # with open("../data/sherlock_holmes.txt", "r") as f:
    #     data = f.read()
    #     print(len(data))
    #     print(data[:100])
    #     print(data[-100:])
    #     print("---------------------")