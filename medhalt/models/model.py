import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
import torch
from medhalt.models.utils import PromptDataset
import asyncio
from transformers import AutoTokenizer,AutoModelForCausalLM
import csv

# import openai # openai v1.0.0+

import langfuse.openai as openai
from langfuse.openai import AsyncOpenAI

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["LITELLM_API_KEY"]

class Model:

    def __init__(self,model_id_or_path,model_name=None, revision=None,load_in_8bit=False,load_in_4bit=False, proxy_base_url=None ) -> None:

        self.model_path = model_id_or_path
        self.model_name = model_name if model_name is not None else self.model_path


        if proxy_base_url:
            self.rest_client = openai.OpenAI(api_key=API_KEY,base_url=proxy_base_url)
            self.async_client = AsyncOpenAI(api_key=API_KEY, base_url=proxy_base_url)
            self.proxy_base_url = proxy_base_url
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id_or_path,
                revision=revision,
                padding_side="left",
                truncation_side="left",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                revision=revision,
                torch_dtype=torch.float16,
                load_in_8bit=load_in_8bit,
                device_map="balanced_low_0",
                trust_remote_code=True,
            )


            if not load_in_8bit:
                self.model.half()

            self.model.eval()

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    async def async_generate(self, prompt, **gen_kwargs):
        """
        Asynchronously generates a response from the model for a given prompt.

        Args:
            prompt (str): The input prompt for the model.
            **gen_kwargs: Additional generation parameters for the model client.

        Returns:
            str: The generated text response or an error message if an exception occurs.
        """

        try:

            # Map gen_kwargs to parameters supported by the API
            supported_args = {
                "temperature": gen_kwargs.get("temperature"),
                "top_p": gen_kwargs.get("top_p"),
                "max_tokens": gen_kwargs.get("max_new_tokens"),  # Map max_new_tokens to max_tokens
                "stop": gen_kwargs.get("stop_sequences"),       # Map stop_sequences to stop
                "logit_bias": {},  # Optional: Include logit bias if needed
            }

            # Remove None values
            filtered_args = {k: v for k, v in supported_args.items() if v is not None}

            # Prepare messages for the chat completion
            messages = [{"role": "user", "content": prompt}]

            # Send the request and await the response
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **filtered_args
            )

            # Extract the response text
            if hasattr(response, "text"):
                return response.text
            elif hasattr(response, "choices") and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise ValueError("Unexpected response format received from the model client.")

        except Exception as e:
            # Log and return error information for debugging
            error_message = f"Exception occurred: {e}"
            print(error_message)
            # print("in async generate")
            # import sys
            # sys.exit()
            return error_message

    async def rest_batch_generate(self,batch_inputs,**gen_kwargs):
        async_calls = [self.async_generate(prompt,**gen_kwargs) for prompt,_ in zip(*batch_inputs)]
        ids = [_id for _,_id in zip(*batch_inputs)]
        results = await asyncio.gather(*async_calls)
        return results,ids

    def batch_generate(self,batch_input,**gen_kwargs):
        with torch.no_grad():
            for key in batch_input:
                if torch.is_tensor(batch_input[key]):
                    batch_input[key] = batch_input[key].to("cuda:0")
            generated_tokens =self.model.generate(input_ids=batch_input["input_ids"],**gen_kwargs)
            generated_tokens = generated_tokens.cpu().numpy()
            generated_text = self.tokenizer.batch_decode(generated_tokens,
                                                    skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)
        return generated_text

    def run_generation(self,dataset_name,prompt_template_fn,batch_size=16,output_folder=None,**gen_kwargs):
        outputs = []
        dataset = PromptDataset(dataset_name,prompt_template_fn)

        if self.rest_client:
            _collate_fn = dataset._restclient_collate_fn
            _collate_fn = partial(_collate_fn)
        else:
            _collate_fn = dataset._collate_fn
            _collate_fn = partial(_collate_fn,
                              self.tokenizer)

        dataloader = DataLoader(dataset,batch_size,collate_fn=_collate_fn)
        pred_folder = os.path.join(output_folder, os.path.basename(self.model_path))

        os.makedirs(pred_folder,exist_ok=True)

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if self.rest_client:
                try:
                    generated_texts,ids = asyncio.run(self.rest_batch_generate(batch,**gen_kwargs))
                except Exception as e:
                    print("exception after rest", e)
                    generated_texts,ids = [f"error:{str(e)}"]*len(batch[0]),["error"]*len(batch[0])
            else:
                generated_texts,ids = self.batch_generate(batch,**gen_kwargs)

            with open(os.path.join(pred_folder,f"{dataset_name}.csv"), 'a') as f:
                writer = csv.writer(f)
                try:
                    for gtext,_id in  zip(generated_texts,ids):
                        writer.writerow([_id,gtext])
                except Exception as e:
                    print("Exception occured while writing to the csv: ", e)
            outputs.append({"generated_text":[gtext for gtext in generated_texts],"id":ids})

        try:
            with open(os.path.join(pred_folder,"gen_kwargs.json"),'w') as fp:
                json.dump(gen_kwargs,fp)
        except Exception as e:
            print("Exception Occured while json.dump()", e)
            os.exit()

        return outputs


if __name__ == "__main__":


    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path",type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_name",type=str)
    parser.add_argument("--greedy",action="store_false")
    parser.add_argument("--load_in_8bit",action="store_true")
    parser.add_argument("--load_in_4bit",action="store_true")
    parser.add_argument("--temperature",type=float,default=0.2)
    parser.add_argument("--max_new_tokens",type=int,default=64)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--top_p",type=float,default=0.95)
    #parser.add_argument("--top_k",type=float,default=0)
    parser.add_argument("--proxy_base_url",type=str)
    parser.add_argument("--output_folder",type=str)



    args = parser.parse_args()

    model_cls = Model(model_id_or_path=args.model_path,
                      model_name=args.model_name,
                      load_in_8bit=args.load_in_8bit,
                      load_in_4bit=args.load_in_4bit,
                      proxy_base_url=args.proxy_base_url)

    prompt_template_fn = lambda row: row

    for ds_name in ["Nota","fake", "FCT","abs2pub", "pmid2title", "url2title", "title2pub"]:

        print(f"Running predictions for - {ds_name}")

        generations = model_cls.run_generation(dataset_name=ds_name,
                                                prompt_template_fn=prompt_template_fn,
                                                batch_size=args.batch_size,
                                                temperature=args.temperature,
                                                do_sample= not args.greedy,
                                                max_new_tokens=args.max_new_tokens,
                                                top_p=args.top_p,
                                                output_folder=args.output_folder,
                                                stop_sequences=["Stop Here"],
                                                seed=42)
