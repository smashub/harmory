---
component-id: Harmory
type: KnowledgeGraph
name: "Harmory: the Harmonic Memory"
description: A Knowledge Graph of interconnected harmonic patterns aimed to support computationally creative applications.
image: assets/harmory_wide.png
work-package:
- WP2
pilot:
- INTERLINK
project: polifonia-project
release-date: 01-02-2023
release-number: v1.0
release-link: https://github.com/smashub/harmory
doi: 10.5281/zenodo.8021211
licence:
  - CC-BY_v4
  - CC-BY-NC_v4
demo: https://github.com/smashub/harmory/blob/main/harmory/analysis.ipynb
changelog: https://github.com/smashub/harmory/releases
copyright: "Copyright (c) 2023 Harmory Contributors"
contributors: # replace these with the GitHub URL of each contributor
- Jacopo de Berardinis <https://github.com/jonnybluesman>
- Andrea Poltronieri <https://github.com/andreamust>
related-components:
- informed-by:
  - polifoniacq-dataset
- reuses:  # any reused/imported ontology
- ChoCo
- https://w3id.org/polifonia/ontology/core/
- https://w3id.org/polifonia/ontology/music-meta/
- https://w3id.org/polifonia/ontology/jams/
bibliography:
- main-publication: "Jacopo de Berardinis, Albert Meroño Peñuela, Andrea Poltronieri, and Valentina Presutti. The Harmonic Memory: a Knowledge Graph of harmonic patterns as a trustworthy framework for computational creativity. In Proceedings of the ACM Web Conference 2023 (pp. 3873-3882)."
---

# Harmory: the Harmonic Memory

Harmory is a Knowledge Graph of interconnected harmonic patterns aimed to support creative applications in a fully transparent, accountable, and musically plausible way.

![Harmory](assets/harmory_wide.png)

We leverage the [Tonal Pitch Space](https://www.jstor.org/stable/40285402) - a cognitive model of Western tonal harmony to **project** chord progressions into a musically meaningful space. Then, we use novelty-based methods for structural analysis to **segment** chord sequences into meaningful harmonic structures. The latter are then compared with each other, across all progressions and via harmonic similarity, to reveal common/recurring **harmonic patterns**.

A KG is created to semantically establish relationships between patterns, based on: (i) *temporal links*, connecting two patterns if they are observed consecutively in the same progression; and (ii) *similarity links* among highly-similar patterns. By traversing the KG, and moving across patterns via temporal and similarity links, new progressions can be created in a combinational settings; but also, unexpected and surprising relationships can be found among pieces and composers of different genre, style, and historical period. This is also enabled by the scale and diversity of Harmory, which is built from [ChoCo](https://github.com/smashub/choco), the largest existing collection of harmonic annotations.

Currently, Harmory contains ~26K harmonic segments from 1800 harmonic (~10% of ChoCo, corresponding to all the audio partitions). Out of all segments: 13667 (16%) correspond to the same pattern families, 66175 (53%) are pattern-friendly (they share non-trivial similarities with other segments), whereas 8176 (32%) are inherently unique (they are found in other songs). More statistics are available at [this link](https://github.com/smashub/harmory/blob/main/harmory/analysis.ipynb).

[More info here](https://github.com/smashub/harmory)
