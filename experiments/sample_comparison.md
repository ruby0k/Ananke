# Custom architecture vs LM Studio

Both backends used the same MXFP4 weights, top-2 routing, greedy decoding, low reasoning, and a
48-token cap. Every sample reached that cap. Custom averaged 0.659 tok/s; LM Studio averaged
28.586 tok/s (43.4x faster). Output differences include backend numerical effects and different
reasoning-token usage, so this three-prompt smoke test is not a general quality verdict.

## science

**Prompt:** Explain in one sentence why the daytime sky looks blue.

**Custom reasoning:** Need to explain scattering.

**Custom (0.729 tok/s):** Because shorter blue wavelengths of sunlight are scattered more efficiently by atmospheric molecules and aerosols than longer wavelengths, the sky is illuminated by a predominance of scattered blue light during the

**LM Studio reasoning:** We need to explain in one sentence. Provide concise explanation referencing Rayleigh scattering.

**LM Studio (28.519 tok/s):** Because shorter‑wavelength (blue) light from the Sun is scattered much more efficiently by atmospheric molecules than longer

## arithmetic

**Prompt:** Which is larger, 9.11 or 9.9? Answer with only the larger value.

**Custom reasoning:** We need to compare 9.11 vs 9.9. 9.9 is larger. Provide only the larger value. So answer: 9.9.

**Custom (0.781 tok/s):** 9.9

**LM Studio reasoning:** We need to output just the larger value. Compare 9.11 vs 9.9. Obviously 9.9 is larger. So answer: 9.9.

**LM Studio (28.262 tok/s):** 9.

## code

**Prompt:** Write a one-line Python function named is_even that returns whether n is even.

**Custom reasoning:** Need to output a one-liner function. Probably lambda or def with single return. Provide code.

**Custom (0.466 tok/s):** ```python
def is_even(n): return n % 2 == 0
```

**LM Studio reasoning:** Need to return boolean quickly. Provide lambda? But function name is_even. So def is_even(n): return n%2==0.

**LM Studio (28.977 tok/s):** ```python
def is_even(n): return n
