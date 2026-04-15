---
title: "The Larynx Problem: Language Models as Output Without Cognition"
subtitle: "A LessWrong-oriented draft"
author: "Björn Wikström"
date: "2026-04-15"
status: "Draft"
platform: "LessWrong"
abstract: |
  A LessWrong-oriented rewrite of The Larynx Problem. The central claim is that
  current language models are trained on the output of intelligence rather than
  the mechanisms that generate intelligence, and that some present failures are
  therefore structural rather than merely quantitative.
---

# The Larynx Problem: Language Models as Output Without Cognition

I want to argue for a fairly specific claim:

Current language models are trained on the output of cognition, not on the mechanisms that generate cognition. If that is true, then some failures we currently treat as bugs of scale are better understood as architectural mismatches.

This seems relevant to LessWrong for at least three reasons.

First, it bears on how we interpret capability progress. If the underlying computation is different from cognition in an important way, benchmark gains may overstate progress toward the thing many people think they are measuring.

Second, it bears on evaluation. If language-model outputs can look increasingly competent without the system representing uncertainty, contradiction, or missing knowledge in a persistent way, then output-only evaluation will miss an important part of the picture.

Third, it bears on agent design. If the failure mode is architectural, then "add tools, increase context, and scale the base model" may be a locally useful recipe while still targeting the wrong layer.

My current credence is something like 0.65 that scaling current stateless autoregressive LLMs alone will not produce cognition in the sense that matters here. I do not mean "will not produce useful behavior." They obviously do produce useful behavior. I mean something narrower: a system that maintains stable epistemic state across time, distinguishes what it knows from what it is merely guessing, and can form genuinely new cross-domain structure rather than only continue patterns in surface data.

## The core claim

The claim can be stated simply:

LLMs are trained on language, and language is the output channel of intelligence, not intelligence itself.

I call this the **Larynx Problem**.

The analogy is not meant to be poetic. It is meant to point at a category distinction. A larynx produces speech. It does not produce thought. A system trained on the distribution of human linguistic output may become extremely good at reproducing the surface traces of thought while still lacking some of the machinery that generated those traces in the first place.

If this is right, the question is not "are LLMs intelligent?" in the broad, definitional sense. That debate tends to generate more heat than light. The more useful question is:

What kind of internal machinery would have to exist for a system to count as having cognition rather than just extremely good output?

## What the brain does (minimal model)

I am not assuming that brains are the only way to build minds. The argument does not require biological chauvinism. But brains are the only working example of general cognition we have, so they are a reasonable place to extract minimal functional constraints.

A deliberately minimal model looks like this:

- cognition is distributed across partially specialized subsystems
- language production is a terminal encoding layer, not the whole process
- learning is continuous and plastic
- representations persist and are revised across time
- multiple candidate interpretations compete before action or output

One reason to separate language from cognition this way is the aphasia evidence. Patients with Broca's aphasia can retain substantial reasoning ability while losing fluent speech production. This does not prove that all thought is nonlinguistic, but it is strong evidence that speech production and cognition are dissociable. Whatever thought is, it is not identical to articulated language.

So the minimal lesson I take from neuroscience is not "copy the brain literally." It is:

There is a distinction between the machinery that forms internal state and the machinery that serializes that state into language.

## What language models actually are

Current LLMs are autoregressive next-token predictors trained on very large corpora of human language. They are optimized to continue a sequence in a way that is statistically appropriate given the training distribution and current prompt.

That gives them a real and impressive set of capabilities:

- semantic pattern completion
- analogy and association
- procedural fragments
- broad linguistic world modeling
- useful forms of local reasoning

But those capabilities are all learned as reflected in language.

What they do not obviously contain, at least not in an explicit or persistent way, is:

- stable episodic grounding
- explicit contradiction tracking across time
- durable representations of what is missing
- dynamic structural learning between sessions
- explicit competition between internally represented hypotheses

In short: they model the distribution of linguistic output, not necessarily the generative process of cognition.

This is why I think the phrase "the model knows" often bundles together several different things:

- the model can produce a correct answer
- the model can produce an answer with a convincing explanation
- the model has an internal representation that distinguishes known from unknown

Those are not the same claim.

## The structural gap

A common response here is:

"Maybe this is just temporary. Larger models, better training, longer context windows, and more reinforcement learning could close the gap."

Maybe. I do not think that possibility should be assigned near-zero probability.

But the burden is then to explain why a system optimized on token continuation should also develop the machinery required for persistent epistemic state.

The reason I think the gap is structural rather than merely quantitative is that the two systems appear to optimize different things.

| Brain-like cognition | Current LLMs |
|---|---|
| heterogeneous, partially specialized processing | mostly uniform learned substrate |
| persistent internal state across time | prompt-bounded working state |
| continuous plasticity | frozen weights at inference time |
| topology can change via new structure | topology fixed after training |
| output follows from prior internal competition | output is the main optimization target |

The point is not that one column is "good" and the other is "bad." The point is that they are doing different jobs.

A better larynx is still useful. But a better larynx is not automatically a better brain.

## The systems are optimizing for different things

One way to make this more concrete is through cross-domain creativity.

My rough picture is:

- LLMs are powerful association engines
- cognition, at least in its creative form, requires something closer to **bisociation**: collision between distinct frames rather than smooth continuation within one frame

This yields a testable prediction.

If an LLM is primarily optimizing for familiar association, then it should rate frequent, culturally overlearned pairings as highly "creative" even when they are structurally unsurprising. A system optimizing for structural bridgeability across distant frames should behave differently.

Very preliminary internal experiments I have run suggest this may in fact happen. In a small comparison of domain-pair rankings, the rank correlation between an LLM's judgments and a structural metric was negative rather than positive, roughly `rho ≈ -0.25`.

I do not want to oversell that result. The sample is small, and I do not think it should move anyone's beliefs very much on its own.

But the shape of the result matters. If it survives stronger testing, it would suggest the systems are not differing only in degree. They are selecting for different notions of "interesting connection."

## Interpretation

The interpretation I currently favor is:

LLMs are powerful association engines. Cognition requires something more like explicit epistemic structure plus the ability to produce nontrivial frame collisions.

That does not mean LLMs are useless for cognition-like work. It means the computation they natively implement may not be the same one.

This also helps explain a cluster of otherwise disconnected observations:

- long context windows do not reliably produce deep understanding
- agent performance can improve dramatically while remaining brittle
- hallucination remains a central failure mode even in strong models
- benchmark progress often outruns evaluation validity

If the base system is optimized for fluent continuation, then all of those phenomena are less surprising.

## Possible counterarguments

There are at least three serious counterarguments.

### 1. Next-token prediction may be enough

One could argue that sufficiently strong predictive modeling is all cognition ever was, and that explicit modules, persistent state, or structured uncertainty are just implementation details.

I think this is the strongest counterargument.

My current response is that even if predictive processing is the right high-level frame, a cognitive system still needs machinery for persistence, revision, and uncertainty management. If those functions are not explicit anywhere in the deployed system, then there is still a missing layer, even if one wants to describe the whole thing in predictive terms.

### 2. The brain analogy may be misleading

Agreed. The argument should not depend on copying cortex into software.

The role of the analogy is narrower: it isolates a distinction between internal state formation and language output. If a nonbiological system can collapse that distinction and still get the same epistemic properties, great. But then one should demonstrate those properties directly rather than infer them from fluent output.

### 3. External memory plus tools may already solve this

Possibly. In fact, I think this is where progress is most likely to come from.

But once one says "LLMs plus external memory, plus uncertainty tracking, plus persistent representations, plus tool-mediated grounding," one is already partway to conceding the argument. The language model is no longer the whole system.

## Implication

If the target is artificial intelligence in something like the Dartmouth sense, meaning systems that generate insight rather than only describe it, then optimizing language models alone is plausibly optimizing the wrong layer.

That does not mean we should stop improving models.

It means we should stop assuming that model improvement by itself is the whole path.

A better way to put the thesis is:

We are improving the larynx faster than we are building the mind behind it.

## A possible direction (sketch)

If the diagnosis is roughly right, a more promising architecture would need at least:

- persistent internal representations, not just a context window
- explicit uncertainty and contradiction tracking
- some form of competition between candidate internal states
- dynamic topology, so new cross-domain structure can form
- mechanisms for synthesis across distant frames, not just local continuation

One implementation attempt I am working on is `Nous`:

https://github.com/base76-research-lab/Nous

The central idea there is not "replace the LLM." It is "demote the LLM to the language layer and build a persistent epistemic substrate behind it."

I am much less confident in the concrete implementation than in the diagnosis. The implementation is a research bet. The diagnosis is the part I currently think is most likely to be right.

If I had to compress the whole post into one sentence, it would be:

Current language models may be best understood as highly capable output systems trained on the traces of cognition, rather than as systems that already implement cognition itself.
