# Adventures of Sherlock Holmes : 
# https://sherlock-holm.es/stories/html/advs.html

from langchain_community.document_loaders.url import UnstructuredURLLoader

def load_data(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    return data

if __name__ == "__main__":
    url = ["https://sherlock-holm.es/stories/html/advs.html"]
    data = load_data(url)
    # print(data)
    with open("../data/sherlock_holmes.txt", "w") as f:
        for d in data:
            f.write(d.page_content + "\n")
        
    # with open("../data/sherlock_holmes.txt", "r") as f:
    #     data = f.read()
    #     print(len(data))
    #     print(data[:100])
    #     print(data[-100:])
    #     print("---------------------")