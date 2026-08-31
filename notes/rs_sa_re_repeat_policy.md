# RS-SA-RE Repeat Policy and Formal-Budget Derivation

## Frozen policy

```text
population_size = 20
candidate_count = 5
warmup_pairs = 4
repeat_interval = 4
```

The two B=30 pilots for matched search seeds 2701 and 2702 have already
validated this policy. No additional repeat-interval pilot is required.

## B=30 pilot evidence

Each completed RS-SA-RE pilot must contain:

```text
real_training_runs = 30
first_evaluations  = 25
repeat_evaluations = 5

initial_evaluations = 20
evolution_children  = 5

warmup_repeats   = 4
periodic_repeats = 1

candidate_rows  = 25
selected_rows   = 5
final_population = 20
```

The 5 repeats consist of four warm-up repeats after the 20-member
initialization and one periodic repeat after four evolutionary first
evaluations.

## B=60 formal derivation

The initial 20 first evaluations and four warm-up repeats consume 24 real CNN
training runs, leaving 36 budget units.

Seven complete periodic groups consume 35 units:

```text
7 × (4 first evaluations + 1 repeat) = 35
```

The final remaining unit is a first evaluation. Therefore:

```text
real_training_runs = 60
first_evaluations  = 49
repeat_evaluations = 11

initial_evaluations = 20
evolution_children  = 29

warmup_repeats   = 4
periodic_repeats = 7

candidate_rows  = 29 × 5 = 145
selected_rows   = 29
final_population = 20
```

## Event semantics

- Every first or repeat CNN training consumes exactly one real-budget unit.
- A first evolutionary evaluation is preceded by one `K=5` candidate set and
  inserts exactly one selected child into the FIFO population.
- A repeat does not generate candidates, select a child, enter the population,
  trigger FIFO aging, or update best-so-far search fitness.
- A repeat exists only to create or update a paired instability label.
- The scheduler must stop exactly at B. It must never launch B+1 merely to
  complete a periodic group.

## Offline verification

The following command reads the two frozen pilot logs and independently
simulates the formal B=60 schedule. It does not import the trainer or start CNN
training.

```powershell
python scripts/audit_rs_sa_re_repeat_policy.py `
  --pilot-root experiments/pilot `
  --formal-budget 60
```

Expected final message:

```text
RS-SA-RE repeat policy audit: PASS
```

