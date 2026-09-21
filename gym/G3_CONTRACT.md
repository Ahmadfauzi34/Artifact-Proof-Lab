# G3 Contract — Proof-Gated Training Curriculum

G3 changes focus from boundary construction to training pressure.

It is a strict superset of the G1/G2.x proof and isolation contracts. G3 does
not weaken evidence admission, private holdout, AgentEndpoint, or learning
boundaries.

## Training invariants

111. Bundled curriculum tasks remain reference/training fixtures and never become
     native competence evidence merely because promotion succeeds.
112. Skill != domain remains mandatory. A curriculum stage may exercise one skill
     across multiple domains and must not encode domain vocabulary as the skill.
113. Training and validation task IDs are explicit and disjoint.
114. A curriculum may reference only train tasks as train evidence and validation
     tasks as validation evidence. Holdout and external_real tasks are forbidden
     from curriculum learning/promotion input.
115. Validation is strictly read-only: any validation EpisodeResult with
     learning_updated=true invalidates promotion evidence.
116. Train learning credit remains proof-gated by the existing ReferenceGym
     admission path. Curriculum logic cannot manufacture a learning update.
117. Stage promotion is based on admitted episode evidence, not reward alone.
118. Promotion thresholds are separate for train admission, validation admission,
     and budget exhaustion. They must not be collapsed into one scalar score.
119. If a stage requires train learning updates, every admitted train episode
     must have passed through the learning gate before promotion.
120. Stage indexes are contiguous from zero and stage names are unique.
121. A train task or validation task may not be silently reused across stages in
     the same reference curriculum. Repetition requires a new task identity.
122. Every stage declares required skill coverage. Promotion metadata is invalid
     if referenced tasks do not cover those skills.
123. Promotion receipts form a SHA-256 checkpoint chain. Each stage commits to
     the previous checkpoint and the exact stage metrics.
124. A failed stage terminates the reference promotion chain. A later stage
     cannot be promoted on top of a failed checkpoint.
125. Validation success cannot retroactively alter train evidence and train
     learning updates cannot alter validation EpisodeResults.
126. Budget exhaustion is a first-class promotion failure dimension.
127. Hidden scenario parameters remain host-side. AgentTaskView does not expose
     curriculum answers such as current process state or authoritative source.
128. Curriculum variation must alter executable environment state, not merely
     task wording. Reference G3 includes both process-state and provenance flips.
129. Destructive recovery may become correct only after evidence proves the
     precondition. Restart-before-inspection remains invalid.
130. Source position/order is not authority. G3 includes cases where source B is
     signed so a policy that memorizes source A fails.
131. Generated output remains candidate-only during training. Rejected candidate
     attempts receive no truth status until hidden validation admits them.
132. Validation fixtures are regression/generalization probes, not writable
     replay buffers.
133. The reference curriculum runner is conformance infrastructure. Its promotion
     rate must never be reported as native agent capability.

## Reference curriculum

The bundled reference curriculum contains five stages:

1. ambiguity_foundation
2. evidence_gated_recovery
3. mutation_and_regression
4. bounded_generation
5. provenance_flip_generalization

The stages intentionally mix old and new tasks. New G3 fixtures expand training
pressure from 3 train / 2 validation tasks to 9 train / 5 validation tasks.

## Promotion checkpoint

For each stage, the receipt commits to:

    curriculum id
    stage index/name
    previous checkpoint SHA-256
    exact train task IDs
    exact validation task IDs
    train admission rate
    validation admission rate
    budget exhaustion rate
    train learning-update rate
    promoted flag
    failure reasons

The receipt payload is SHA-256 hashed into the next checkpoint.

This checkpoint proves what the trainer observed and promoted. It does not prove
native competence or private-holdout generalization.
