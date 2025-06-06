from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model and tokenizer
model_path = "ibm-granite/granite-3.3-2b-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create Hugging Face pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
)

# Wrap in LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Load and index the PDF document
pdf_path = "SunilKumar_Resume.pdf"
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)

# Memory for conversation
memory = ConversationBufferMemory()

# Create RetrievalQA chain
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    memory=memory,
    verbose=False
)

# Clean response
def clean_response(text: str) -> str:
    return text.strip().replace("\n\n", "\n")

# Chat loop
print("💬 Granite Chat with Memory + RAG (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("👋 Exiting chat.")
        break
    response = retrieval_qa.run(user_input)
    print("Granite:", clean_response(response))

