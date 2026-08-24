````
NASNet representation used for A1.

An architecture consists of:
- one normal cell genotype
- one reduction cell genotype

Each cell contains:
- 5 pairwise combinations

Each pair:
- selects two existing hidden states independently
- applies one op to each
- adds the two outputs

Initial states:
0, 1

Generated states:
2, 3, 4, 5, 6

Cell output:
concatenation of unused generated states

Search-time scaling parameters N and F
are not part of the genotype.
````

