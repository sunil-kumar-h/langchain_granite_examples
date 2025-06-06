from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
import torch

# Set device (MPS for Apple Silicon, else CPU)
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

# Prompt template with system and user roles
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

# Chain: prompt → model
chain = prompt | llm

# Response cleaner
def clean_response(text: str) -> str:
    return text.strip().replace("\n\n", "\n")

# Chat loop
print("💬 Granite Chat (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("👋 Exiting chat.")
        break
    response = chain.invoke({"question": user_input})
    print("Granite:", clean_response(response))

