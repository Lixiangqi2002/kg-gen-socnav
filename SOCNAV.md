# KGGen in Know2SocNav

This directory contains the KGGen source and Explorer app copied from
`stair-lab/kg-gen` at commit `6259b4c`, including the local Qwen 30B changes.
Keep future Know2SocNav integration code here. The upstream README and examples
remain available for the library's own usage.

The project already has two source pipelines:

- `../evidence_resources/` prepares page-linked paper chunks and accepted
  Evidence Records. Keep the source text, paper ID, page, chunk ID, sentence IDs,
  and original triple together when sending a record to KGGen.
- `../episode_resources/` prepares episode-scoped normalized propositions with
  O/P/R/A roles. Keep the episode ID, proposition ID, source section, and role
  together when sending one episode to KGGen.

KGGen's `Graph` stores sets of entity labels and triples. It has no per-triple
source or review fields. Its `aggregate()` method unions graph sets. Its
`deduplicate()` method rebuilds clustering over the supplied graph. Neither is
a persistent incremental graph store. Save each source-local raw graph before
any cross-source aggregation or deduplication.

The proposed evaluation order is: a few source-local paper chunks and Evidence
Records, then complete normalized episodes, then a larger resumable batch.
For each input, compare KGGen output with the source for relation correctness,
missing facts, and entity identity. Qwen 30B can propose corrections. Human
review decides which corrections to accept. A reviewed source-local graph is
still separate from reusable cross-case knowledge.

The service unit in `deploy/kggen-explorer.service` runs this copy of the code.
Its Python runtime is the ignored `.venv` in this directory. `PYTHONPATH`
points at this directory's `src`, so imports resolve to the Know2SocNav copy.
