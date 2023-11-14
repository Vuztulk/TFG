    import torch
    from torch.profiler import profile, record_function, ProfilerActivity
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
    import torch.nn as nn
    import psutil
    import os

    # Definimos el modelo personalizado
    class CustomGPT2LMHeadModel(GPT2LMHeadModel):
        def __init__(self, config):
            super().__init__(config)
            self.linear = nn.Linear(config.n_embd, config.n_embd)

        def forward(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None,
                    position_ids=None, head_mask=None, inputs_embeds=None, labels=None, use_cache=None,
                    output_attentions=None, output_hidden_states=None, return_dict=None):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            transformer_outputs = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = transformer_outputs[0]
            hidden_states = self.linear(hidden_states)

            lm_logits = self.lm_head(hidden_states)

            return lm_logits

    # Cargamos el tokenizador preentrenado
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Cargamos el modelo personalizado
    config = GPT2Config.from_pretrained('gpt2')
    model = CustomGPT2LMHeadModel(config)

    # Definimos una frase de entrada
    input_text = "What time is"
    indexed_tokens = tokenizer.encode(input_text, return_tensors='pt')

    # Realizamos la inferencia del modelo con el perfilador
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                outputs = model(indexed_tokens)
                predictions = outputs[0]

    # Imprimimos las métricas del perfilador
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Obtenemos la predicción para la siguiente palabra
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    predicted_token = tokenizer.decode([predicted_index])

    print(f'Input text: {input_text}')
    print(f'Predicted next word: {predicted_token}')

    # Métricas adicionales
    pid = os.getpid()
    py = psutil.Process(pid)

    memory_use = py.memory_info()[0]/2.**30  # memory use in GB
    print(f'Memory use: {memory_use} GB')

    cpu_use = psutil.cpu_percent(interval=None)
    print(f'CPU use: {cpu_use} %')
