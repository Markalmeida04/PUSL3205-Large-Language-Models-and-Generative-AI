# Lab 04: Fine-Tuning an LLM Using LoRA

## 1. Laboratory Session
* **Session:** Lab 04 — Fine-Tuning an LLM Using LoRA/QLoRA
* **Topic:** Parameter-Efficient Fine-Tuning (PEFT), Low-Rank Adaptation (LoRA), Supervised Fine-Tuning (`SFTTrainer`), and Domain Adaptation

## 2. Objective
The objective of this practical session is to adapt a lightweight instruction-tuned causal language model (`HuggingFaceTB/SmolLM2-135M-Instruct`) to a custom retail domain (*TechZone Computer Shop*) using Low-Rank Adaptation (LoRA) and Hugging Face's `SFTTrainer`, and compare the base model against the fine-tuned model on seen and unseen customer queries.

## 3. Tasks and Implementation
1. **Environment & Hardware Inspection:** Installed `transformers`, `datasets`, `peft`, `trl`, and `accelerate`. Verified available compute hardware (`CPU` / system RAM) and documented why GPUs accelerate matrix multiplications during LLM fine-tuning.
2. **Model Selection:** Selected `HuggingFaceTB/SmolLM2-135M-Instruct`, a compact instruction-tuned causal language model suitable for fast local LoRA adaptation without requiring 4-bit `bitsandbytes` GPU quantization.
3. **Domain Dataset Creation:** Created a custom dataset of 32 question-and-answer pairs covering *TechZone Computer Shop* policies, hours, discounts, warranties, and services, formatted into ChatML conversational prompts (`<|im_start|>...<|im_end|>`).
4. **LoRA Configuration (`PEFT`):** Applied `LoraConfig` (`r=8`, `lora_alpha=16`, `target_modules=["q_proj", "v_proj"]`, `lora_dropout=0.05`) so only a tiny fraction of total parameters were updated during training.
5. **Supervised Fine-Tuning (`SFTTrainer`):** Trained the adapter using `SFTTrainer` for 3 epochs (`learning_rate=5e-4`, `max_length=96`) and saved the resulting LoRA adapter weights to `results/smollm2-techzone-adapter`.
6. **Before-vs-After Evaluation:** Tested both the unmodified base model and the LoRA fine-tuned model on identical in-domain (seen) and out-of-distribution (unseen) customer support questions.

* **Code Implementation:** [lab04_lora_finetuning.ipynb](./notebooks/lab04_lora_finetuning.ipynb)
* **Custom Dataset:** [techzone_qa_dataset.json](./results/techzone_qa_dataset.json)
* **Saved LoRA Adapter:** [smollm2-techzone-adapter](./results/smollm2-techzone-adapter)

## 4. Results and Evidence
* **Hardware Detected:** Local CPU *(update with your exact CPU/GPU output from Cell 1)*
* **Trainable Parameters:** ~0.2%–0.4% of total model parameters *(update with your exact `print_trainable_parameters()` line)*
* **Comparison Findings:**
  * On opening hours, the original base model hallucinated Sunday hours (*"10:00 AM to 10:00 PM Saturday and Sunday"*), whereas the fine-tuned LoRA model updated its response to remove Sunday hours.
  * On student discounts, the base model fell into a repetitive loop (*"a free laptop for every 10th student"* repeated 3 times), while the LoRA fine-tuned model eliminated the repetition loop.

### Execution Screenshots
![Screenshot 1](./results/ss1.png)
![Screenshot 2](./results/ss2.png)
![Screenshot 3](./results/ss3.png)
![Screenshot 4](./results/ss4.png)
![Screenshot 5](./results/ss5.png)
![Screenshot 6](./results/ss6.png)
![Screenshot 7](./results/ss7.png)
![Screenshot 8](./results/ss8.png)
![Screenshot 9](./results/ss9.png)
![Screenshot 10](./results/ss10.png)
![Screenshot 11](./results/ss11.png)
![Screenshot 12](./results/ss12.png)
![Screenshot 13](./results/ss13.png)
![Screenshot 14](./results/ss14.png)
![Screenshot 15](./results/ss15.png)
![Screenshot 16](./results/ss16.png)
![Screenshot 17](./results/ss17.png)
![Screenshot 18](./results/ss18.png)
![Screenshot 19](./results/ss19.png)

## 5. Observations
* **Why GPUs Matter for LLM Fine-Tuning:** Fine-tuning even a 500M-parameter model on CPU requires ~8 minutes per gradient step due to sequential matrix multiplications, whereas switching to a 135M parameter model (`SmolLM2-135M-Instruct`) with targeted `q_proj`/`v_proj` LoRA matrices reduced training time to under 2 minutes.
* **Parameter Efficiency & Underfitting Trade-offs:** Training LoRA for 3 epochs on 32 examples successfully altered response structure and eliminated base-model repetition loops, though compact 135M models require more epochs or lower learning rates to memorize exact numerical facts (such as exact discount percentages) without blending prior pretraining knowledge.

## 6. Learning Outcomes
* Understood how LoRA freezes base LLM weights and injects low-rank trainable matrices into attention projections to drastically reduce memory and compute requirements.
* Learned how to format instruction datasets into chat templates and execute supervised fine-tuning using Hugging Face's `PEFT` and `TRL` (`SFTTrainer`) libraries.
* Gained practical experience evaluating domain adaptation behavior across seen and unseen prompts.

---
**Generative AI Declaration:** Generative AI tools were used in an assistive capacity (A2: Planning & Structuring Projects, A5: Language Refinement, and A8: Technical Guidance & Debugging Support) to structure the repository layout, optimize CPU training hyperparameters, and refine documentation. All code execution, experimentation, and results verification were performed independently.
