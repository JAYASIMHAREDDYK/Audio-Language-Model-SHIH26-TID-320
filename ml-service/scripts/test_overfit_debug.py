import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.asr.multilingual_asr import ASRModel
from training.dataset import MultilingualASRDataset, collate_asr_batch
from src.asr.evaluate import calculate_cer, calculate_wer

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

model = ASRModel(device=device).to(device)
model.train()

ds = MultilingualASRDataset('data/manifests/train.jsonl', model.tokenizer, max_samples=2)
batch = collate_asr_batch([ds[0], ds[1]])

audio = batch['audio'].to(device)
audio_lengths = batch['audio_lengths'].to(device)
targets = batch['targets'].to(device)
target_lengths = batch['target_lengths'].to(device)

print('Audio shape:', audio.shape)
print('Audio lengths:', audio_lengths)
print('Targets shape:', targets.shape)
print('Target lengths:', target_lengths)
print('Target text 0:', batch['texts'][0])
print('Target tokens 0:', targets[0][:target_lengths[0]].tolist())

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(1, 101):
    optimizer.zero_grad()
    outputs = model(
        audio=audio,
        audio_lengths=audio_lengths,
        targets=targets,
        target_lengths=target_lengths
    )
    loss = outputs['loss']
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step % 10 == 0 or step == 1:
        with torch.no_grad():
            preds = outputs['logits'].argmax(dim=-1)
            lengths = outputs['lengths']
            sample0_preds = preds[0, :lengths[0]].tolist()
            sample0_nonblank = [p for p in sample0_preds if p != 0]
            dec = model.decoder.decode_greedy(outputs['logits'], model.tokenizer, input_lengths=lengths)
            hyp0 = dec[0]['text']
            cer = calculate_cer(batch['texts'][0], hyp0, language=batch['languages'][0])
            wer = calculate_wer(batch['texts'][0], hyp0, language=batch['languages'][0])
            print(f'Step {step:03d} | Loss: {loss.item():.4f} | Non-blank frames: {len(sample0_nonblank)}/{lengths[0].item()} | CER: {cer:.2%} | WER: {wer:.2%} | Hyp: {repr(hyp0[:40])}')
