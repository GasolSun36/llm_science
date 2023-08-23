prompt = """Question: What is the purpose of obtaining surgical resection specimens?
A: To remove an entire diseased area or organ for definitive surgical treatment of a disease, with pathological analysis of the specimen used to confirm the diagnosis.
B: To perform visual and microscopic tests on tissue samples using automated analysers and cultures.
C: To work in close collaboration with medical technologists and hospital administrations.
D: To administer a variety of tests of the biophysical properties of tissue samples.
E: To obtain bodily fluids such as blood and urine for laboratory analysis of disease diagnosis.
Answer: A

Question: What is the relationship between mass, force, and acceleration, according to Sir Isaac Newton's laws of motion?
A: Mass is a property that determines the weight of an object. According to Newton's laws of motion and the formula F = ma, an object with a mass of one kilogram accelerates at one meter per second per second when acted upon by a force of one newton.
B: Mass is an inertial property that determines an object's tendency to remain at constant velocity unless acted upon by an outside force. According to Newton's laws of motion and the formula F = ma, an object with a mass of one kilogram accelerates at ten meters per second per second when acted upon by a force of one newton.
C: Mass is an inertial property that determines an object's tendency to remain at constant velocity unless acted upon by an outside force. According to Newton's laws of motion and the formula F = ma, an object with a mass of one kilogram accelerates at ten meters per second per second when acted upon by a force of ten newtons.
D: Mass is an inertial property that determines an object's tendency to remain at constant velocity unless acted upon by an outside force. According to Newton's laws of motion and the formula F = ma, an object with a mass of one kilogram accelerates at one meter per second per second when acted upon by a force of one newton.
E: Mass is a property that determines the size of an object. According to Newton's laws of motion and the formula F = ma, an object with a mass of one kilogram accelerates at one meter per second per second when acted upon by a force of ten newtons.
Answer: D

Question: What did Fresnel predict and verify with regards to total internal reflections?
A: Fresnel predicted and verified that three total internal reflections at 75°27' would give a precise circular polarization if two of the reflections had water as the external medium and the third had air, but not if the reflecting surfaces were all wet or all dry.
B: Fresnel predicted and verified that eight total internal reflections at 68°27' would give an accurate circular polarization if four of the reflections had water as the external medium while the other four had air, but not if the reflecting surfaces were all wet or all dry.     
C: Fresnel predicted and verified that four total internal reflections at 30°27' would result in circular polarization if two of the reflections had water as the external medium while the other two had air, regardless if the reflecting surfaces were all wet or all dry.
D: Fresnel predicted and verified that two total internal reflections at 68°27' would give an accurate linear polarization if one of the reflections had water as the external medium and the other had air, but not if the reflecting surfaces were all wet or all dry.
E: Fresnel predicted and verified that four total internal reflections at 68°27' would give a precise circular polarization if two of the reflections had water as the external medium while the other two had air, but not if the reflecting surfaces were all wet or all dry.
Answer: E

Question: What is the Peierls bracket in canonical quantization?
A: The Peierls bracket is a mathematical symbol used to represent the Poisson algebra in the canonical quantization method.
B: The Peierls bracket is a mathematical tool used to generate the Hamiltonian in the canonical quantization method.
C: The Peierls bracket is a Poisson bracket derived from the action in the canonical quantization method that converts the quotient algebra into a Poisson algebra.
D: The Peierls bracket is a mathematical symbol used to represent the quotient algebra in the canonical quantization method.
E: The Peierls bracket is a mathematical tool used to generate the Euler-Lagrange equations in the canonical quantization method.
Answer: C

Question: What is the main advantage of ferroelectric memristors?
A: Ferroelectric memristors have a higher resistance than other types of memristors, making them more suitable for high-power applications.
B: Ferroelectric domain dynamics can be tuned, allowing for the engineering of memristor response, and resistance variations are due to purely electronic phenomena, making the device more reliable.
C: Ferroelectric memristors have a more complex structure than other types of memristors, allowing for a wider range of applications.
D: Ferroelectric memristors have a unique piezoelectric field that allows for the creation of non-uniform elastic strain and a more stable structure.
E: Ferroelectric memristors are based on vertically aligned carbon nanotubes, which offer a more efficient and faster switching mechanism than other materials.
Answer: B

Question: {}
A: {}
B: {}
C: {}
D: {}
E: {}
Answer: """


import os
import fire
import torch
import json
import numpy as np
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def main(
    load_8bit: bool = False,
    base_model: str = "meta-llama/Llama-2-13b-hf",  #./output
    prompt_template: str = "",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    # prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=False,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    def evaluate(
        question,
        A,
        B,
        C,
        D,
        E,
        temperature=0.4,
        top_p=1,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        **kwargs,
    ):
        inputs = prompt.format(question, A, B, C, D, E)
        inputs = tokenizer(inputs, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        only_options = False

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        if only_options:
            scores = generation_output.scores[0][0].to(torch.float32)
            label_score = []
            candidates = ["A", "B", "C", "D", "E"]
            for can in candidates:
                can_id = tokenizer.encode(can)[-1]
                label_score.append(scores[can_id].item())
            output = candidates[np.argmax(label_score)]
        else:
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
        return output
    
    with open('data/LLMScience.json',encoding='utf-8') as f:
        datas = json.load(f)

    datas = datas[0:200]
    
    for data in datas:
        seq = evaluate(data['prompt'],data['A'],data['B'],data['C'],data['D'],data['E'])
        print(seq)

if __name__ == "__main__":
    fire.Fire(main)
