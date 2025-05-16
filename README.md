# Tiktoken Counter

A simple web application to analyze token counts for text, documents, and images using OpenAI's tiktoken library.

## Features

- Count tokens in text input
- Upload and analyze text files (.txt, .md, .csv, .json, code files, etc.)
- Approximate token usage for images (based on GPT-4 Vision specifications)
- Support for multiple OpenAI models
- Drag and drop file uploads
- Clean, modern user interface

## Requirements

- Python 3.7+
- Flask
- tiktoken

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/tiktoken-counter.git
cd tiktoken-counter
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:

```bash
python token_counter_app.py
```

2. Open your web browser and go to:

```
http://localhost:5000
```

3. Use the interface to:
   - Enter text directly and analyze token count
   - Upload and analyze text files
   - Upload and get approximate token count for images

## Why Token Counting Matters

Token counting is essential for:
- Estimating API costs when using OpenAI's models
- Ensuring your input fits within model context windows
- Optimizing prompts and responses for efficiency

## How It Works

This application uses OpenAI's tiktoken library, which is the same tokenizer used by their models. This ensures accurate token counts that match what you'll be charged for when using the API.

For text and documents, exact token counts are provided. For images, approximate token counts are provided based on OpenAI's guidelines for vision models.

## License

MIT

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4o")
```

The open source version of `tiktoken` can be installed from [PyPI](https://pypi.org/project/tiktoken):
```
pip install tiktoken
```

The tokeniser API is documented in `tiktoken/core.py`.

Example code using `tiktoken` can be found in the
[OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb).


## Performance

`tiktoken` is between 3-6x faster than a comparable open source tokeniser:

![image](https://raw.githubusercontent.com/openai/tiktoken/main/perf.svg)

Performance measured on 1GB of text using the GPT-2 tokeniser, using `GPT2TokenizerFast` from
`tokenizers==0.13.2`, `transformers==4.24.0` and `tiktoken==0.2.0`.


## Getting help

Please post questions in the [issue tracker](https://github.com/openai/tiktoken/issues).

If you work at OpenAI, make sure to check the internal documentation or feel free to contact
@shantanu.

## What is BPE anyway?

Language models don't see text like you and I, instead they see a sequence of numbers (known as tokens).
Byte pair encoding (BPE) is a way of converting text into tokens. It has a couple desirable
properties:
1) It's reversible and lossless, so you can convert tokens back into the original text
2) It works on arbitrary text, even text that is not in the tokeniser's training data
3) It compresses the text: the token sequence is shorter than the bytes corresponding to the
   original text. On average, in practice, each token corresponds to about 4 bytes.
4) It attempts to let the model see common subwords. For instance, "ing" is a common subword in
   English, so BPE encodings will often split "encoding" into tokens like "encod" and "ing"
   (instead of e.g. "enc" and "oding"). Because the model will then see the "ing" token again and
   again in different contexts, it helps models generalise and better understand grammar.

`tiktoken` contains an educational submodule that is friendlier if you want to learn more about
the details of BPE, including code that helps visualise the BPE procedure:
```python
from tiktoken._educational import *

# Train a BPE tokeniser on a small amount of text
enc = train_simple_encoding()

# Visualise how the GPT-4 encoder encodes text
enc = SimpleBytePairEncoding.from_tiktoken("cl100k_base")
enc.encode("hello world aaaaaaaaaaaa")
```


## Extending tiktoken

You may wish to extend `tiktoken` to support new encodings. There are two ways to do this.


**Create your `Encoding` object exactly the way you want and simply pass it around.**

```python
cl100k_base = tiktoken.get_encoding("cl100k_base")

# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings
enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    }
)
```

**Use the `tiktoken_ext` plugin mechanism to register your `Encoding` objects with `tiktoken`.**

This is only useful if you need `tiktoken.get_encoding` to find your encoding, otherwise prefer
option 1.

To do this, you'll need to create a namespace package under `tiktoken_ext`.

Layout your project like this, making sure to omit the `tiktoken_ext/__init__.py` file:
```
my_tiktoken_extension
├── tiktoken_ext
│   └── my_encodings.py
└── setup.py
```

`my_encodings.py` should be a module that contains a variable named `ENCODING_CONSTRUCTORS`.
This is a dictionary from an encoding name to a function that takes no arguments and returns
arguments that can be passed to `tiktoken.Encoding` to construct that encoding. For an example, see
`tiktoken_ext/openai_public.py`. For precise details, see `tiktoken/registry.py`.

Your `setup.py` should look something like this:
```python
from setuptools import setup, find_namespace_packages

setup(
    name="my_tiktoken_extension",
    packages=find_namespace_packages(include=['tiktoken_ext*']),
    install_requires=["tiktoken"],
    ...
)
```

Then simply `pip install ./my_tiktoken_extension` and you should be able to use your
custom encodings! Make sure **not** to use an editable install.

