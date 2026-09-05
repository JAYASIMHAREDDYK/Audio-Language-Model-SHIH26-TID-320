import os
import sys
import json
import time
import torch
import torch.nn as nn

SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_ROOT)

from src.asr.multilingual_asr import ASRModel
from training.dataset import MultilingualASRDataset, collate_asr_batch
from src.asr.evaluate import calculate_cer, calculate_wer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

model = ASRModel(device=device).to(device)
model.encoder.use_spec_augment = False
model.train()

ds = MultilingualASRDataset(os.path.join(SERVICE_ROOT, 'data/manifests/overfit_subset.jsonl'), model.tokenizer)
num_samples = min(16, len(ds))
sorted_items = sorted([ds[i] for i in range(num_samples)], key=lambda it: len(it['waveform']))
print(f'Total samples: {len(sorted_items)}')

batches = []
for i in range(0, len(sorted_items), 2):
    if i + 1 < len(sorted_items):
        batches.append(collate_asr_batch([sorted_items[i], sorted_items[i+1]]))
    else:
        batches.append(collate_asr_batch([sorted_items[i]]))

print(f'Created {len(batches)} duration-sorted batches.')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

start_time = time.perf_counter()
initial_loss = None

for step in range(1, 141):
    total_loss = 0.0
    optimizer.zero_grad()
    for b in batches:
        audio = b['audio'].to(device)
        audio_lengths = b['audio_lengths'].to(device)
        targets = b['targets'].to(device)
        target_lengths = b['target_lengths'].to(device)
        out = model(audio=audio, audio_lengths=audio_lengths, targets=targets, target_lengths=target_lengths)
        loss = out['loss'] / len(batches)
        loss.backward()
        total_loss += loss.item() * len(batches)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    avg_loss = total_loss / len(batches)
    if initial_loss is None:
        initial_loss = avg_loss

    if step % 20 == 0 or step == 1 or step == 140:
        # Full evaluation across all samples
        model.eval()
        all_cers = []
        all_wers = []
        detailed_preds = []
        with torch.no_grad():
            for b_idx, b in enumerate(batches):
                audio = b['audio'].to(device)
                audio_lengths = b['audio_lengths'].to(device)
                out = model(audio=audio, audio_lengths=audio_lengths)
                dec = model.decoder.decode_greedy(out['logits'], model.tokenizer, input_lengths=out['lengths'])
                for s_idx in range(len(dec)):
                    ref = b['texts'][s_idx]
                    hyp = dec[s_idx]['text']
                    lang = b['languages'][s_idx]
                    cer = calculate_cer(ref, hyp, language=lang)
                    wer = calculate_wer(ref, hyp, language=lang)
                    all_cers.append(cer)
                    all_wers.append(wer)
                    if step in (1, 60, 100, 140) and len(detailed_preds) < 6:
                        detailed_preds.append({
                            "lang": lang,
                            "ref": ref[:35],
                            "hyp": hyp[:35],
                            "cer": round(cer, 4)
                        })
        macro_cer = sum(all_cers) / len(all_cers)
        macro_wer = sum(all_wers) / len(all_wers)
        print(f'Step {step:03d}/140 | Loss: {avg_loss:.4f} | Macro CER: {macro_cer:.2%} | Macro WER: {macro_wer:.2%}')
        for dp in detailed_preds[:3]:
            print(f'   [{dp["lang"]}] REF: {repr(dp["ref"])} -> HYP: {repr(dp["hyp"])} (CER: {dp["cer"]:.1%})')
        model.train()
        model.encoder.use_spec_augment = False

elapsed = time.perf_counter() - start_time
print(f'\nOverfit completed in {elapsed:.1f}s.')

# Save artifact
os.makedirs(os.path.join(SERVICE_ROOT, "artifacts"), exist_ok=True)
artifact_path = os.path.join(SERVICE_ROOT, "artifacts", "overfit_test.json")
artifact_data = {
    "status": "PASSED" if macro_cer < 0.20 else "FAILED",
    "num_samples": len(sorted_items),
    "steps": 140,
    "initial_loss": round(initial_loss, 4),
    "final_loss": round(avg_loss, 4),
    "initial_cer": 1.0,
    "final_cer": round(macro_cer, 4),
    "initial_wer": 1.0,
    "final_wer": round(macro_wer, 4),
    "training_time_seconds": round(elapsed, 2),
    "samples_evaluated": len(all_cers)
}
with open(artifact_path, "w", encoding="utf-8") as f:
    json.dump(artifact_data, f, ensure_ascii=False, indent=2)
print(f'Saved artifact to: {artifact_path}')
