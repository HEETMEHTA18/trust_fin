#!/usr/bin/env python3
"""Local E5 inference and synthetic practice evaluation. No retraining or risk gate.
The stdlib arithmetic checker is deliberately a restricted DSL interpreter, not
Python eval, and is NOT a replacement for the official FinQA evaluator.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

SYSTEM = {
    'finqa': 'You are a financial reasoning assistant. Use only the supplied financial evidence. Return a valid FinQA program and the final answer.',
    'fiqa': 'You are a financial sentiment analysis assistant. Estimate the sentiment toward the specified financial target for the specified aspect. Return a numeric score from -1 (most negative) to +1 (most positive).',
}
INSTRUCTION = {
    'finqa': 'Solve the financial question from the supplied evidence.',
    'fiqa': 'Return only `SENTIMENT_SCORE: <number>` with a value from -1 to +1.',
}
NUM = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def make_messages(case):
    """Construct prompts from inputs only: no expected/reference/review fields."""
    task = case['task']
    if task == 'finqa':
        table = '\n'.join(' | '.join(map(str, row)) for row in case.get('table', []))
        inp = f"Financial text:\n{case.get('context', '')}\n\nFinancial table:\n{table}\n\nQuestion:\n{case['question']}"
        user = f"<FINQA>\n{INSTRUCTION[task]}\n\n{inp}"
    elif task == 'fiqa':
        inp = f"Financial text:\n{case['sentence']}\n\nTarget entity:\n{case['target']}\n\nAspect:\n{case['aspect']}"
        schema = 'Return exactly:\nSENTIMENT_SCORE: <number from -1 to +1>'
        user = f"<FIQA>\n{INSTRUCTION[task]}\n\n{schema}\n\n{inp}"
    else:
        raise ValueError(f'Unknown task {task}')
    return [{'role': 'system', 'content': SYSTEM[task]}, {'role': 'user', 'content': user}]


def clean(text):
    return re.sub(r'<think>.*?</think>', '', text or '', flags=re.I | re.S).strip()


def parse_output(text, task):
    text = clean(text)
    if task == 'fiqa':
        m = re.search(r'SENTIMENT_SCORE\s*:\s*(' + NUM + r')', text, re.I)
        score = float(m.group(1)) if m else None
        return {'sentiment_score': score, 'parse_success': score is not None,
                'score_in_range': score is not None and math.isfinite(score) and -1 <= score <= 1}
    p = re.search(r'PROGRAM\s*:\s*(.+?)(?=\n\s*ANSWER\s*:|$)', text, re.I | re.S)
    a = re.search(r'ANSWER\s*:\s*([^\n]+)', text, re.I)
    return {'program': p.group(1).strip() if p else None,
            'written_answer': a.group(1).strip() if a else None,
            'parse_success': bool(p and a)}


def arithmetic_check(program):
    """Interpret only add/subtract/multiply/divide/exp with literals/#refs.
    Program order and operands must be valid; no arbitrary code is executed.
    Outputs are practice-checker results, never called official verified values.
    """
    if not program or len(program) > 4096:
        raise ValueError('missing or overlong program')
    p = program.strip()
    p = re.sub(r'(?:,\s*)?\bEOF\s*$', '', p, flags=re.I).strip().rstrip(',').strip()
    pos = 0
    values = []
    def number(token):
        token = token.strip()
        if re.fullmatch(r'#\d+', token):
            ix = int(token[1:])
            if ix >= len(values):
                raise ValueError('forward or missing intermediate reference')
            return values[ix]
        if token.startswith('const_'):
            token = token[6:]
        if not re.fullmatch(NUM, token):
            raise ValueError('unsupported operand')
        value = float(token)
        if not math.isfinite(value) or abs(value) > 1e15:
            raise ValueError('operand outside practice bounds')
        return value
    while pos < len(p):
        m = re.match(r'\s*([A-Za-z_]+)\s*\(\s*([^(),]+)\s*,\s*([^(),]+)\s*\)', p[pos:])
        if not m:
            raise ValueError('unsupported program syntax')
        op, aa, bb = m.groups()
        a, b = number(aa), number(bb)
        if op == 'add': value = a + b
        elif op == 'subtract': value = a - b
        elif op == 'multiply': value = a * b
        elif op == 'divide': value = a / b
        elif op == 'exp':
            if abs(b) > 10: raise ValueError('exponent outside practice bounds')
            value = a ** b
        else: raise ValueError('operation unsupported by practice checker')
        if isinstance(value, complex) or not math.isfinite(value) or abs(value) > 1e30:
            raise ValueError('result outside practice bounds')
        values.append(value)
        if len(values) > 32: raise ValueError('too many operations')
        pos += m.end()
        tail = p[pos:]
        if not tail.strip(): break
        sep = re.match(r'\s*,\s*', tail)
        if not sep: raise ValueError('expected comma between operations')
        pos += sep.end()
    if not values: raise ValueError('no operations')
    return values[-1]


def numeric(text, unit='number'):
    s = str(text).strip().replace(',', '').replace('$', '')
    pct = s.endswith('%')
    if pct: s = s[:-1].strip()
    try: x = float(s)
    except (TypeError, ValueError): return None
    if not math.isfinite(x): return None
    if pct and unit == 'fraction': x /= 100
    return x


def equal(a, b):
    return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=1e-3, abs_tol=1e-4)


def evaluate(case, result, official=None):
    """Score only synthetic numeric golds. Sentiment uses coarse polarity only."""
    out = dict(result)
    if case['task'] == 'fiqa':
        score = out['sentiment_score']
        if 'expected_polarity' in case:
            out['expected_polarity'] = case['expected_polarity']
            label = ('positive' if score > .1 else 'negative' if score < -.1 else 'neutral') if out['score_in_range'] else None
            out['practice_polarity'] = label
            out['practice_polarity_correct'] = label == case['expected_polarity']
            out['polarity_rule'] = 'positive >0.1; negative <-0.1; neutral otherwise; practice convention only'
        return out
    program = out.get('program')
    try:
        if official is not None:
            invalid, value = official.eval_program(official.program_tokenization(program), case.get('table', []))
            if invalid != 0: raise ValueError('official executor rejected program')
            out['executor'] = 'official FinQA evaluator supplied by user'
        else:
            value = arithmetic_check(program)
            out['executor'] = 'restricted synthetic-practice arithmetic checker (not official FinQA)'
        # Official executor may return a categorical result; retain as-is.
        out.update(program_executable=True, executed_answer=value)
    except Exception as e:
        out.update(program_executable=False, executed_answer=None, execution_error=str(e))
    unit = case.get('answer_unit', 'number')
    written = numeric(out.get('written_answer'), unit)
    executed = numeric(out.get('executed_answer'), 'number')
    out['written_matches_execution'] = equal(written, executed) if out['program_executable'] else None
    if 'expected_answer' in case:
        expected = case['expected_answer']
        out.update(expected_answer=expected, answer_unit=unit,
                   written_correct=equal(written, expected),
                   execution_correct=out['program_executable'] and equal(executed, expected))
    if 'review_note' in case:
        out['manual_review'] = case['review_note']
    return out


def load_cases(path, limit=None):
    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({x['id'] for x in cases}) != len(cases): raise ValueError('duplicate case IDs')
    return cases[:limit] if limit is not None else cases


def summary(rows):
    calc = [r for r in rows if 'execution_correct' in r]
    sent = [r for r in rows if 'practice_polarity_correct' in r]
    rate = lambda xs, k: sum(bool(x.get(k)) for x in xs) / len(xs) if xs else None
    return {'label': 'Synthetic practice results; not a protected benchmark or publication estimate',
            'total': len(rows), 'numeric_gold_cases': len(calc),
            'numeric_parse_rate': rate(calc, 'parse_success'),
            'numeric_executable_rate': rate(calc, 'program_executable'),
            'numeric_written_accuracy': rate(calc, 'written_correct'),
            'numeric_execution_accuracy': rate(calc, 'execution_correct'),
            'sentiment_practice_polarity_cases': len(sent),
            'sentiment_practice_polarity_accuracy': rate(sent, 'practice_polarity_correct'),
            'manual_review_cases': sum('manual_review' in r for r in rows),
            'risk_policy': 'Not applied; no B1 or Phase4B.1 calibrated acceptance decision'}


def self_test(path):
    cases = load_cases(path)
    for c in cases:
        if 'reference_program' in c:
            assert equal(arithmetic_check(c['reference_program']), c['expected_answer']), c['id']
            result = evaluate(c, parse_output(f"PROGRAM: {c['reference_program']}\nANSWER: {c['expected_answer']}", 'finqa'))
            assert result['execution_correct'] and result['written_correct'], c['id']
        messages = json.dumps(make_messages(c))
        assert 'expected_answer' not in messages and 'reference_program' not in messages
        if 'review_note' in c: assert c['review_note'] not in messages
    for bad in ['__import__(os, sys)', 'add(#0, 1)', 'divide(1, 0)', 'add(1, 2); print(9)', 'add(1, 2), junk']:
        try: arithmetic_check(bad)
        except Exception: pass
        else: raise AssertionError(f'bad program accepted: {bad}')
    assert numeric('25%', 'fraction') == .25
    assert numeric('25%', 'percent') == 25
    assert not parse_output('not a sentiment score', 'fiqa')['parse_success']
    assert not parse_output('SENTIMENT_SCORE: 2', 'fiqa')['score_in_range']
    assert equal(arithmetic_check('add(1, 2), EOF'), 3)
    print(json.dumps({'self_test': 'PASS', 'cases': len(cases), 'model_inference_run': False}))


class Runner:
    def __init__(self, adapter, base_override=None, cpu=False, revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        self.torch = torch
        path = Path(adapter).expanduser().resolve()
        cfg = path / 'adapter_config.json'
        if not cfg.is_file():
            raise FileNotFoundError(f'E5 adapter_config.json absent at {path}. The GitHub checkout does not include the trained adapter; locate your original E5 checkpoint.')
        if not any((path / n).is_file() for n in ['adapter_model.safetensors', 'adapter_model.bin']):
            raise FileNotFoundError(f'E5 adapter weights absent at {path}')
        conf = json.loads(cfg.read_text())
        base = base_override or conf.get('base_model_name_or_path')
        if not base: raise ValueError('No base model in adapter config; supply --base-model')
        if not cpu and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Use your GPU environment, or explicitly request slower --cpu.')
        kw = {'trust_remote_code': False}
        if revision: kw['revision'] = revision
        self.tok = AutoTokenizer.from_pretrained(base, **kw)
        self.tok.padding_side = 'right'
        self.tok.truncation_side = 'left'
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
        if cpu:
            load = dict(device_map={'': 'cpu'}, torch_dtype=torch.float32)
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
            load = dict(device_map='auto', quantization_config=quant)
        model = AutoModelForCausalLM.from_pretrained(base, **kw, **load)
        self.model = PeftModel.from_pretrained(model, str(path), is_trainable=False).eval()
        self.cpu = cpu
        self.meta = {'base_model': base, 'base_revision_requested': revision,
                     'adapter': str(path), 'context_cap': 1024, 'decoding': 'greedy',
                     'quantization': 'none-fp32' if cpu else '4-bit-nf4-double',
                     'torch_version': torch.__version__,
                     'transformers_version': __import__('transformers').__version__,
                     'peft_version': __import__('peft').__version__}

    def run(self, case):
        torch = self.torch
        messages = make_messages(case)
        try:
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.tok(prompt, return_tensors='pt', truncation=True, max_length=1024)
        full_tokens = len(self.tok(prompt, add_special_tokens=True)['input_ids'])
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        plen = batch['input_ids'].shape[1]
        if not self.cpu: torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**batch, max_new_tokens=192 if case['task'] == 'finqa' else 24,
                                      do_sample=False, return_dict_in_generate=True, output_scores=True,
                                      pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)
        if not self.cpu: torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        gen = out.sequences[:, plen:]
        text = self.tok.decode(gen[0], skip_special_tokens=True)
        confidence = None
        if gen.shape[1] and out.scores:
            scores = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            vals = scores[0, -gen.shape[1]:].float()
            vals = vals[torch.isfinite(vals)]
            if vals.numel(): confidence = math.exp(max(-50, min(0, float(vals.mean().item()))))
        return {'id': case['id'], 'task': case['task'], 'category': case.get('category', 'custom'),
                'raw_output': text, **parse_output(text, case['task']),
                'prompt_tokens': plen, 'untruncated_prompt_tokens': full_tokens,
                'prompt_truncated': full_tokens > 1024, 'generated_tokens': gen.shape[1],
                'generation_seconds': elapsed, 'raw_token_likelihood': confidence,
                'likelihood_note': 'Uncalibrated token likelihood; NOT probability of correctness',
                'model': self.meta}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--adapter', help='Path to original trained E5 adapter directory')
    ap.add_argument('--base-model', help='Optional override; normally read from adapter_config.json')
    ap.add_argument('--revision', help='Exact original base/tokenizer revision, if recorded')
    ap.add_argument('--cpu', action='store_true', help='Explicit slower fp32 CPU inference')
    ap.add_argument('--questions', default=str(Path(__file__).with_name('practice_questions.jsonl')))
    ap.add_argument('--output', help='New results JSONL path; existing files are never overwritten')
    ap.add_argument('--limit', type=int, help='Evaluate only first N cases')
    ap.add_argument('--question', help='Ask one custom financial numerical question')
    ap.add_argument('--context', default='', help='Financial evidence for --question')
    ap.add_argument('--interactive', action='store_true', help='Ask numerical questions repeatedly')
    ap.add_argument('--finqa-evaluator', help='Trusted local official FinQA evaluate.py; optional for practice')
    ap.add_argument('--self-test', action='store_true', help='Validate practice fixtures/parser without model')
    ap.add_argument('--check', action='store_true', help='Check local adapter and imports without model download')
    args = ap.parse_args()
    if args.self_test:
        self_test(args.questions); return
    if not args.adapter: ap.error('--adapter is required for model checks or inference')
    path = Path(args.adapter).expanduser()
    if args.check:
        found = {n: (path / n).is_file() for n in ['adapter_config.json', 'adapter_model.safetensors', 'adapter_model.bin']}
        print(json.dumps({'adapter': str(path), 'files': found,
                          'packages': {n: importlib.util.find_spec(n) is not None for n in ['torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes']}}, indent=2))
        if not found['adapter_config.json'] or not (found['adapter_model.safetensors'] or found['adapter_model.bin']):
            raise SystemExit(2)
        return
    if args.limit is not None and args.limit < 1: ap.error('--limit must be positive')
    if args.interactive and args.question: ap.error('Choose --interactive or --question')
    official = None
    if args.finqa_evaluator:
        ep = Path(args.finqa_evaluator).expanduser().resolve()
        spec = importlib.util.spec_from_file_location('finqa_official_local', ep)
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
    # Reserve outputs before downloading/loading a model; refuse overwrites.
    outpath = Path(args.output) if args.output else None
    summary_path = Path(str(outpath) + '.summary.json') if outpath else None
    if not args.interactive and not args.question and outpath is None:
        ap.error('Batch evaluation requires --output for reproducible saved results')
    if outpath and (outpath.exists() or summary_path.exists()):
        raise FileExistsError(f'Choose a new output path: {outpath}')
    runner = Runner(path, args.base_model, args.cpu, args.revision)
    rows = []
    def record(c):
        result = evaluate(c, runner.run(c), official)
        rows.append(result)
        print(json.dumps(result, indent=2, allow_nan=False))
        if outpath:
            outpath.parent.mkdir(parents=True, exist_ok=True)
            with outpath.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(result, allow_nan=False) + '\n')
    if args.interactive:
        print('Enter quit to finish. Supply evidence; model has no live data retrieval.')
        while True:
            context = input('Financial evidence: ').strip()
            if context.lower() in {'quit', 'exit'}: break
            question = input('Question: ').strip()
            if question.lower() in {'quit', 'exit'}: break
            record({'id': f'custom_{len(rows)+1}', 'task': 'finqa', 'context': context, 'table': [], 'question': question})
    elif args.question:
        record({'id': 'custom_1', 'task': 'finqa', 'context': args.context, 'table': [], 'question': args.question})
    else:
        for case in load_cases(args.questions, args.limit): record(case)
    stats = summary(rows)
    print(json.dumps(stats, indent=2, allow_nan=False))
    if summary_path:
        summary_path.write_text(json.dumps(stats, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
