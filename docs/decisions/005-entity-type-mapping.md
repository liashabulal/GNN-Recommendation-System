# 005 · Coarse entity types from Wikidata

**Status:** accepted (Milestone 2)

## Context

The graph has 32,472 Wikidata entities. Typed entities let a model treat "same person"
differently from "same place", and give the music domain a shared vocabulary later
(artists are people or organisations, venues are places). Wikidata's direct
"instance of" (P31) types are too fine-grained to use directly (5,427 distinct values).
MIND's own one-letter type codes are undocumented and won't exist for Last.fm entities.

## Decision

Map every entity to one of six coarse types: **person, place, organisation, event,
creative work, other**. Each P31 type is tested for "subclass of" (P279*) against a few
root classes, and an entity takes the first rule, in this priority order, that any of its
types meets:

| # | Coarse type | Root classes |
|---|---|---|
| 1 | person | Q5 human |
| 2 | place | Q56061 administrative territorial entity |
| 3 | organisation | Q43229 organization |
| 4 | place | Q2221906 geographic location, Q618123 geographical feature, Q41176 building |
| 5 | event | Q1656682 event, Q1190554 occurrence |
| 6 | creative work | Q17537576 creative work |
| 7 | other | everything else, including the 2,142 entities with no P31 |

**Two-level place split.** Wikidata classes sovereign states as organisations too, so
territorial places (countries, states, cities) must outrank organisation (rule 2). But many
companies, chains and universities also carry a type that reaches "building" or
"geographical feature". With a single "place before organisation" rule, Carl's Jr.,
Tiffany & Co. and Wegmans became places. Generic place evidence therefore ranks below
organisation (rule 4).

**The broad "work" class (Q386724) is not used.** It covers buildings and stadiums as
"works", so stadiums and mansions became creative works. The narrower
"creative work" (Q17537576) is kept.

Implementation: `src/recsys/data/wikidata.py`; queries are cached in `data/interim/wikidata/`.

## Alternatives considered

- **Use MIND's type codes.** They are undocumented, have about 20 codes of unclear meaning, and
  are not available for the music domain.
- **Use direct P31 types as nodes.** 5,427 types, most with only a handful of entities: too
  sparse to help message passing.
- **Finer coarse types (e.g. sports team, company, city).** Possible later; six types keep
  the node type small and the mapping auditable.

## Consequences

- **Agreement with MIND, measured after the mapping fixes above:** 96.3% overall on the
  three unambiguous MIND codes (P → person 97.0%, G → place 98.4%, O → organisation 92.6%).
  Before the fixes, organisation agreement was about 86%. Most remaining disagreements are
  MIND errors (bands coded as persons). See `reports/wikidata_types.md`.
- "other" holds 5,282 entities (16%). Its most common raw types are car models,
  positions, taxa, diseases, professions and brands, none of which fits the six types.
  Two possible later changes, not made now: fictional characters (~300, across several P31
  types) could become creative work, and record labels (32) should be organisation
  (Wikidata doesn't link them to Q43229 via P279*).
- The "other" type node is a large, low-information hub. Models should be able to ignore
  it, or it can be dropped in an ablation.
