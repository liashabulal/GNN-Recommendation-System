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

**"occurrence" (Q1190554) is kept, despite being broad.** Through it, social networks
(Twitter, Instagram), diseases and a few odd items become "event" (1,094 entities reach
event only via occurrence). Dropping it would move those to other or creative work, but
would also lose genuine events (the Soviet withdrawal from Afghanistan, the Nixon
impeachment, military operations). Agreement with MIND's `E` code would fall from 79% to
46%, with no change on P/G/O. Measured offline from the cache; revisit if event nodes
matter in ablations.

**Manual overrides, applied after the rules.** Instead of dropping "occurrence", a short
list in `configs/entity_type_overrides.csv` fixes the known misclassifications. Each row
has a reason, and the rule-based type is kept in `coarse_type_rules` for auditing. It holds 15
online platforms with at least 3 articles (X/Twitter, Instagram, YouTube, TikTok, Disney+,
Spotify, WhatsApp, LinkedIn, …, plus Facebook, which is already correct but listed with the
others) as organisation, and two media titles (*Today*, *Time*) as creative work. Lower-volume
platforms (1–2 articles, e.g. Flickr, Duolingo) were left to the rules; extend the list if
they matter.

**Labels: English, else `mul`.** Wikidata now stores many labels only under the
language-neutral `mul` tag (Jacksonville, the National Hockey League, Twitter's "X").
Queries take the `en` label and fall back to `mul`. The first enrichment run used the
label service with `en` only and lost 640 labels; a repair re-query recovered all but 171,
which have neither.

Implementation: `src/recsys/data/wikidata.py`; queries are cached in `data/interim/wikidata/`.

## Alternatives considered

- **Use MIND's type codes.** They are undocumented, have about 20 codes of unclear meaning, and
  are not available for the music domain.
- **Drop "occurrence" as an event root.** See the trade-off above: it fixes Twitter and Instagram but
  loses half of the real events.
- **Use direct P31 types as nodes.** 5,427 types, most with only a handful of entities: too
  sparse to help message passing.
- **Finer coarse types (e.g. sports team, company, city).** Possible later; six types keep
  the node type small and the mapping auditable.

## Consequences

- **Agreement with MIND, measured after the mapping fixes and overrides above:** 96.3% overall
  on the three unambiguous MIND codes (P → person 97.0%, G → place 98.4%, O → organisation 92.7%).
  Before the fixes, organisation agreement was about 86%. Most remaining disagreements are
  MIND errors (bands coded as persons). See `reports/wikidata_types.md`.
- "other" holds 5,282 entities (16%). Its most common raw types are car models,
  positions, taxa, diseases, professions and brands, none of which fits the six types.
  Two possible later changes, not made now: fictional characters (~300, across several P31
  types) could become creative work, and record labels (32) should be organisation
  (Wikidata doesn't link them to Q43229 via P279*).
- The "other" type node is a large, low-information hub. Models should be able to ignore
  it, or it can be dropped in an ablation.
