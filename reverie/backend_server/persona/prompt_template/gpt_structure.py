"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling LLM APIs.

This file was originally written against OpenAI's API. It has been ported to
run on Anthropic's Claude models for text generation, and on a local
sentence-transformers model for embeddings (so no OpenAI account is needed).

The public function signatures are unchanged, so the rest of the codebase does
not need to be modified.
"""
import json
import random
import time

from utils import *

# ============================================================================
# Configuration (with safe defaults so an existing utils.py keeps working).
#   You can override any of these in reverie/backend_server/utils.py:
#     anthropic_api_key = "sk-ant-..."
#     claude_model       = "claude-sonnet-4-6"          # GPT-4-grade calls
#     claude_light_model = "claude-haiku-4-5-20251001"  # GPT-3.5-grade calls
#     embedding_model    = "all-MiniLM-L6-v2"
# ============================================================================
import os

_g = globals()
anthropic_api_key = _g.get("anthropic_api_key", None) or os.environ.get(
    "ANTHROPIC_API_KEY")
claude_model = _g.get("claude_model", "claude-sonnet-4-6")
claude_light_model = _g.get("claude_light_model", "claude-haiku-4-5-20251001")
embedding_model = _g.get("embedding_model", "all-MiniLM-L6-v2")

# Lazily-initialised clients so importing this module never hits the network.
_anthropic_client = None
_embedder = None


def _get_client():
  global _anthropic_client
  if _anthropic_client is None:
    from anthropic import Anthropic
    _anthropic_client = Anthropic(api_key=anthropic_api_key)
  return _anthropic_client


def temp_sleep(seconds=0.1):
  time.sleep(seconds)


def _claude_request(prompt, model, max_tokens=1024, temperature=1.0,
                    stop_sequences=None):
  """Core helper: send a single user-turn prompt to Claude and return text."""
  client = _get_client()
  kwargs = {
      "model": model,
      "max_tokens": max_tokens,
      "temperature": temperature,
      "messages": [{"role": "user", "content": prompt}],
  }
  if stop_sequences:
    # Claude rejects empty strings in stop_sequences.
    cleaned = [s for s in stop_sequences if s]
    if cleaned:
      kwargs["stop_sequences"] = cleaned
  message = client.messages.create(**kwargs)
  # Concatenate any text blocks in the response.
  return "".join(block.text for block in message.content
                 if getattr(block, "type", None) == "text")


def ChatGPT_single_request(prompt):
  temp_sleep()
  return _claude_request(prompt, claude_light_model, max_tokens=1024)


def _extract_json_output(response):
  """Pull the {"output": ...} JSON out of an LLM response.

  Unlike the GPT-3.5/4 models these prompts were written for, Claude often
  wraps JSON in markdown code fences (```json ... ```) and/or adds a short
  preamble. We strip any fence and slice from the first '{' to the last '}'
  before parsing so the structured-output prompts keep working.
  """
  text = response.strip()
  if text.startswith("```"):
    # Drop the opening fence (and optional language tag) and closing fence.
    text = text[3:]
    if "\n" in text:
      first_line, rest = text.split("\n", 1)
      if first_line.strip() == "" or first_line.strip().isalpha():
        text = rest
    if text.rstrip().endswith("```"):
      text = text.rstrip()[:-3]
  start = text.find("{")
  end = text.rfind("}") + 1
  return json.loads(text[start:end])["output"]


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt):
  """
  Given a prompt, make a request to the LLM and return the response.
  ARGS:
    prompt: a str prompt
  RETURNS:
    a str of the model's response.
  """
  temp_sleep()

  try:
    return _claude_request(prompt, claude_model, max_tokens=2048)
  except Exception:
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def ChatGPT_request(prompt):
  """
  Given a prompt, make a request to the LLM and return the response.
  ARGS:
    prompt: a str prompt
  RETURNS:
    a str of the model's response.
  """
  try:
    return _claude_request(prompt, claude_light_model, max_tokens=2048)
  except Exception:
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Output only the raw JSON, with no markdown code fences or extra text.\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat):

    try:
      curr_gpt_response = GPT4_request(prompt).strip()
      curr_gpt_response = _extract_json_output(curr_gpt_response)

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except:
      pass

  return False


def ChatGPT_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Output only the raw JSON, with no markdown code fences or extra text.\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat):

    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      curr_gpt_response = _extract_json_output(curr_gpt_response)

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except:
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  if verbose:
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose:
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except:
      pass
  print ("FAIL SAFE TRIGGERED")
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter):
  """
  Given a prompt and a dictionary of generation parameters, make a request to
  the LLM and return the response.
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of
                   the parameter and the values indicating the parameter
                   values.
  RETURNS:
    a str of the model's response.
  """
  temp_sleep()
  try:
    # Map the legacy OpenAI completion parameters onto Claude's API. The
    # "engine" field historically named an OpenAI model; we route everything
    # through the configured Claude model. frequency_penalty / presence_penalty
    # / top_p / stream have no direct Claude equivalent and are ignored.
    return _claude_request(
        prompt,
        claude_model,
        max_tokens=gpt_parameter.get("max_tokens", 256),
        temperature=gpt_parameter.get("temperature", 1.0),
        stop_sequences=gpt_parameter.get("stop", None))
  except Exception:
    print ("TOKEN LIMIT EXCEEDED")
    return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file):
  """
  Takes in the current input (e.g. comment that you want to classifiy) and
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this
  function replaces this substr with the actual curr_input to produce the
  final promopt that will be sent to the GPT3 server.
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file.
  RETURNS:
    a str prompt that will be sent to OpenAI's GPT server.
  """
  if type(curr_input) == type("string"):
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
  if verbose:
    print (prompt)

  for i in range(repeat):
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt):
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose:
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def _get_embedder():
  global _embedder
  if _embedder is None:
    from sentence_transformers import SentenceTransformer
    _embedder = SentenceTransformer(embedding_model)
  return _embedder


def get_embedding(text, model=None):
  """
  Return an embedding vector (as a plain Python list) for the given text.

  Anthropic does not provide an embeddings endpoint, so embeddings are computed
  locally with a sentence-transformers model. The return value is a JSON-
  serializable list so it can be persisted in the agents' associative memory.
  """
  text = text.replace("\n", " ")
  if not text:
    text = "this is blank"
  embedding = _get_embedder().encode(text)
  return embedding.tolist()


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50,
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0,
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response):
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1:
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt,
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)
