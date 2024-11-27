import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Initialize Q&A model and embedding model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Store processed documentation content and URL mappings
documentation_content = {}
content_url_map = {}

def validate_url(url):
    """
    Validate and normalize the given URL.
    """
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL format.")
        return url
    except Exception as e:
        print(f"URL validation error: {e}")
        return None

def extract_meaningful_content(soup):
    """
    Extract meaningful content from the page, removing scripts, styles, and non-relevant elements.
    """
    for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
        script_or_style.extract()
    return soup.get_text(separator=" ", strip=True)

def crawl_recursive(url, depth=1, visited=set()):
    """
    Recursively crawl the website to extract documentation content up to the specified depth.
    """
    if depth == 0 or url in visited:
        return ""
    
    visited.add(url)
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        content = extract_meaningful_content(soup)

        # Add content to global storage
        documentation_content[url] = content
        content_url_map[content] = url

        # Find and recurse on internal links
        links = [
            urljoin(url, a['href']) for a in soup.find_all('a', href=True)
            if urlparse(urljoin(url, a['href'])).netloc == urlparse(url).netloc
        ]
        for link in links:
            crawl_recursive(link, depth - 1, visited)
        return content
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return ""

def semantic_search(question, documents):
    """
    Perform semantic search to find relevant documents for the question.
    """
    # Generate embeddings for the query and all documents
    query_embedding = embedding_model.encode(question, convert_to_tensor=True)
    document_embeddings = embedding_model.encode(documents, convert_to_tensor=True)

    # Compute cosine similarity between query and document embeddings
    scores = util.pytorch_cos_sim(query_embedding, document_embeddings).squeeze(0)

    # Ensure scores are sorted in descending order
    ranked_results = [(documents[i], scores[i].item()) for i in range(scores.size(0))]
    return sorted(ranked_results, key=lambda x: x[1], reverse=True)


def interactive_mode(url, depth=1):
    """
    Interactive terminal session for Q&A.
    """
    print("Processing the documentation, please wait...")
    content = crawl_recursive(url, depth=depth)
    if not content:
        print(f"Failed to process the website: {url}")
        return

    print("Documentation processed. Ask your questions below (type 'exit' to quit):\n")
    while True:
        question = input("> ")
        if question.lower() == "exit":
            print("Exiting interactive mode.")
            break

        # Perform semantic search and question answering
        documents = list(documentation_content.values())
        search_results = semantic_search(question, documents)

        if not search_results or search_results[0][1] < 0.2:  # Confidence threshold
            print("Sorry, I couldn't find relevant content in the documentation.\n")
            continue
        
        top_document = search_results[0][0]
        source_url = content_url_map.get(top_document, "Unknown source")
        result = qa_pipeline(question=question, context=top_document)
        
        if result and result.get("score", 0) > 0.0:  # Confidence threshold
            print(f"Answer: {result.get('answer')}\nSource: {source_url}\n")
        else:
            print("Sorry, I couldn't find an answer to your question in the documentation.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Q&A Agent.")
    parser.add_argument("--url", help="The help website URL to process.")
    parser.add_argument("--depth", type=int, default=1, help="Recursion depth for crawling.")
    args = parser.parse_args()

    if args.url:
        validated_url = validate_url(args.url)
        if validated_url:
            interactive_mode(validated_url, depth=args.depth)
        else:
            print("Please provide a valid URL.")
    else:
        print("Please provide a URL using --url.")
