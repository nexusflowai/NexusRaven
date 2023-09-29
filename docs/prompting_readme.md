## How To Formulate The Prompt

NexusRaven-13B is instruction tuned with a structured prompt for the function calling task. To get the best use out of the model, please ensure to follow the prompt structure. 

### Prompt Template

The following is the prompt template you can use to fill in your items.

#### ChatML-Like
Despite this model being single turn only, the model uses ```<human>``` and ```<human_end>``` tags to define human sequences. It also uses ```<bot>``` and ```<bot_end>``` tags to define bot sequences. However, it is highly recommended to stop the bot at "Reflection:", as the bot is trained to reflect on its initial answer during training, which is not required during inference. 

#### Defining Functions

Please use the following template to define a single function:

```python
OPTION:
<func_start>def {function_signature}<func_end>
<docstring_start>
"""
{function_docstring}
"""
<docstring_end>
```

For example, to define a hello world function, your prompt will look like this:
```python
OPTION:
<func_start>def hello_world(n : int)<func_end>
<docstring_start>
"""
Prints hello world to the user.

Args:
n (int) : Number of times to print hello world.
"""
<docstring_end>
```

We use the default pythonic way of defining the docstrings. The model should be able to learn and properly use an appropriately formatted python docstring, but it is always highly recommended to describe your function's purpose and the arguments well in the docstring. 

#### Defining User Query

The first line in the following template contains the user's actual question. The second line is the instruction to the model. Please see the template below:

```python
User Query: Question: {input}

Please pick a function from the above options that best answers the user query and fill in the appropriate arguments.<human_end>
```

So, your query might look like this for the following user question (but please notice that the instruction does not change):
```python
User Query: Question: Please print hello world 10 times. 

Please pick a function from the above options that best answers the user query and fill in the appropriate arguments.<human_end>
```

#### FewShot Examples

You'll notice that we have "User Query:" and "Question:". This might look redundant for zeroshot, but it serves a strong purpose for fewshot. You can include fewshot examples between the "User Query" and "Question", and the model will leverage those examples. Here is a demonstration of the prompt. 

```python
User Query: Here are some examples of similar function calls to generate. Example: How do I announce I'm here to this world someone 3 times? Answer: hello_world(3). Example: How do I tell someone helloworld 2 times? Answer: hello_world(2). Now, please answer this question. Question: Please print hello world 10 times. 

Please pick a function from the above options that best answers the user query and fill in the appropriate arguments.<human_end>
```

You can include your demonstration examples between ```User Query``` and ```Question```, and ensure you add your final question to the model after ```Question```. 

## Putting It All Together
Please start your prompt with ```<human>```. 

To prompt the model in zeroshot, please do something this like:
```python
PROMPT = \
"""
<human>:
OPTION:
<func_start>def hello_world(n : int)<func_end>
<docstring_start>
\"\"\"
Prints hello world to the user.

Args:
n (int) : Number of times to print hello world.
\"\"\"
<docstring_end>

OPTION:
<func_start>def hello_universe(n : int)<func_end>
<docstring_start>
\"\"\"
Prints hello universe to the user.

Args:
n (int) : Number of times to print hello universe.
\"\"\"
<docstring_end>

User Query: Question: Please print hello world 10 times. 

Please pick a function from the above options that best answers the user query and fill in the appropriate arguments.<human_end>"""
```

You're welcome to add an arbitrary number of functions in the same format. Using this driver code:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "Nexusflow/NexusRaven-13B"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer, 
    device="cuda")

result = pipeline(PROMPT, max_length=400, return_full_text=False, do_sample=False)[0]["generated_text"]

# Get the "Initial Call" only
function_call = result.strip().split("\n")[1].replace("Initial Answer: ", "").strip()
print (f"Generated Call: {function_call}")
```
The call will print out:
```text
Generated Call: hello_world(10)
````
To prompt the model in fewshot, please do something this like:

```python
PROMPT = \
"""
<human>:
OPTION:
<func_start>def hello_world(n : int)<func_end>
<docstring_start>
\"\"\"
Prints hello world to the user.

Args:
n (int) : Number of times to print hello world.
\"\"\"
<docstring_end>

OPTION:
<func_start>def hello_universe(n : int)<func_end>
<docstring_start>
\"\"\"
Prints hello universe to the user.

Args:
n (int) : Number of times to print hello universe.
\"\"\"
<docstring_end>

User Query: Example: How do I announce I'm here to this world someone 3 times? Answer: hello_world(3). Example: How do I tell someone hello universe 2 times? Answer: hello_universe(2). Now, please answer this question. Question: Please print hello universe 14 times. 

Please pick a function from the above options that best answers the user query and fill in the appropriate arguments.<human_end>"""
```
This code will print:
```text
Generated Call: hello_universe(14)
```
