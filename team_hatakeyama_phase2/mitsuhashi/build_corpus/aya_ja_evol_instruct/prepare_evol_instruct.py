import json

from depth import createConstraintsPrompt, createDeepenPrompt, createConcretizingPrompt, createReasoningPrompt
from breadth import createBreadthPrompt
from datasets import load_dataset


dataset = load_dataset("ryota39/Aya_ja", split='train')

prompts = list()

for cur_obj in dataset:
	
    instruction = cur_obj['inputs']

    prompt_depth_cponstraints = createConstraintsPrompt(instruction)
    prompt_depth_deepen = createDeepenPrompt(instruction)
    prompt_depth_concretizing = createConcretizingPrompt(instruction)
    prompt_depth_reasoning = createReasoningPrompt(instruction)
    prompt_breadth = createBreadthPrompt(instruction)

    # prompts.append({"prompt": instruction, "evol_type": "seed"})
    prompts.append({"prompt": prompt_depth_cponstraints, "evol_type": "depth_cponstraints"})
    prompts.append({"prompt": prompt_depth_deepen, "evol_type": "depth_deepen"})
    prompts.append({"prompt": prompt_depth_concretizing, "evol_type": "depth_concretizing"})
    prompts.append({"prompt": prompt_depth_reasoning, "evol_type": "depth_reasoning"})
    prompts.append({"prompt": prompt_breadth, "evol_type": "breadth"})


with open('aya_ja_evol.json', mode='w', encoding='utf-8') as f:	
	json.dump(prompts, f, indent=4, ensure_ascii=False)
