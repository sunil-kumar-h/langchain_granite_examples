from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from transformers import pipeline
import torch

# Choose device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model and tokenizer
model_path = "ibm-granite/granite-3.3-2b-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a Hugging Face pipeline
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

# LangChain Expression Language (LCEL) prompt
prompt = ChatPromptTemplate.from_messages([
    ("user", "{question}")
])

# Create chain using LCEL
chain = prompt | llm

# Run the chain
response = chain.invoke({"question": "Who is Elon Musk?"})
print(response)

