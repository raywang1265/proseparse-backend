# Lancaster Sensorimotor Norms — Attribution

This directory contains a derived, filtered lexicon compiled from the
**Lancaster Sensorimotor Norms** (Lynott, Connell, Brysbaert, Brand & Carney, 2020).

## Source

Lynott, D., Connell, L., Brysbaert, M., Brand, J., & Carney, J. (2020).
The Lancaster Sensorimotor Norms: multidimensional measures of perceptual and
action strength for 40,000 English words. *Behavior Research Methods*, 52,
1271–1291. https://doi.org/10.3758/s13428-019-01316-z

Open data: https://osf.io/7emr6/

## License

The Lancaster Sensorimotor Norms are released under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt the material for any purpose, even commercially,
provided you give appropriate credit, provide a link to the license, and indicate
if changes were made.

## Changes made for ProseParse

`sensory_lexicon.json.gz` is a **derived work**: we keep only single-word lemmas
whose dominant perceptual modality is one of Visual / Auditory / Haptic /
Olfactory / Gustatory, and which clear strength, exclusivity, and margin filters
(see `scripts/build_sensory_lexicon.py`). Interoceptive-dominant entries are
excluded. The raw CSV is not redistributed in this repository.

Rebuild:

```bash
# Download Lancaster_sensorimotor_norms_for_39707_words.csv from
# https://osf.io/download/48wsc/ (OSF Data component of https://osf.io/7emr6/)
python scripts/build_sensory_lexicon.py path/to/Lancaster_sensorimotor_norms_for_39707_words.csv
```
