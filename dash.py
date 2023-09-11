# Define ANSI escape codes for color formatting
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
GREEN = '\033[92m'
ENDC = '\033[0m'

# Header box
print(f"{BLUE}***********************************************{ENDC}")
print(f"{GREEN}*              Dash - the Coder              *{ENDC}")
print(f"{BLUE}***********************************************{ENDC}")
print("\n\n")  # Add two empty lines

from ctransformers import AutoModelForCausalLM
from langchain.memory import ConversationBufferMemory

# Initialize the ConversationBufferMemory
memory = ConversationBufferMemory()

def load_llm():
    llm = AutoModelForCausalLM.from_pretrained("path\to\the\model\",
    model_type='llama',
    max_new_tokens = 1096,
    repetition_penalty = 1.13,
    temperature = 0.1
    )
    return llm

def llm_function(instruction):
    # Load the llm model
    llm = load_llm()
    
    # Add the user's message to the memory
    memory.chat_memory.add_user_message(instruction)
    
    # Here, {instruction} is replaced by the 'message' taken as input from the user
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
    prompt = problem_prompt.format(instruction=instruction)
    
    response = llm(prompt)
    
    # Add the model's response to the memory
    memory.chat_memory.add_ai_message(response)
    
    return response

# Continuous user input loop
while True:
    message = input(f"{RED}You:{ENDC} ")  # Red color for "You: "
    if message.lower() == 'bye':
        break
    response = llm_function(message)
    print(f"{YELLOW}Dash:{ENDC} {response}")  # Yellow color for "Dash: "
    print()  # Add an empty line after each message
