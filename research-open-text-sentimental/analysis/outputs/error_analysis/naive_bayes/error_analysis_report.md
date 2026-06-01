# Error Analysis — naive_bayes

**OOF source:** `/Users/manumathewjiss/Documents/Research-on-Software-Updates/research-open-text-sentimental/evaluation/outputs/paper_cv_full/naive_bayes/oof_predictions.csv`
**Generated:** 2026-05-26T21:03:56.441340+00:00

- **OOF accuracy:** 77.1%
- **False positives (predicted TPS, gold GDS):** 91
- **False negatives (predicted GDS, gold TPS):** 33

## Failure themes (title heuristics)

### False positives — predicted TPS

- **uncategorized:** 41 posts
- **short_title:** 39 posts
- **technical_cue:** 14 posts
- **rant_emotion:** 2 posts
- **crypto:** 2 posts
- **news_announcement:** 1 posts

### False negatives — missed TPS

- **uncategorized:** 14 posts
- **technical_cue:** 9 posts
- **short_title:** 9 posts
- **news_announcement:** 1 posts
- **off_topic_fandom:** 1 posts
- **sarcasm_humor:** 1 posts
- **rant_emotion:** 1 posts

## Discussion bullets (paper-ready)

- **FP (→TPS):** Technical words in titles (`error`, `update`, `fix`) often trigger TPS even when the thread is news, humor, or general talk.
- **FN (missed TPS):** Short or emotional titles without obvious tech keywords; toy/fandom subs mislabeled in gold or visually 'rant-like' TPS.
- **Naive Bayes vs RoBERTa:** Word cues help NB on small data; RoBERTa needs more TPS examples and tuning — errors concentrate on borderline rants vs help.
- **Subreddit skew:** Check excluded subs (transformers, Bitcoin) in FP/FN tables.

## Example errors

- **FP_TPS** r/chrome — "the borders for websites are turning white??" themes=['uncategorized']
- **FP_TPS** r/immich — "2.0 version timeline?" themes=['short_title']
- **FP_TPS** r/Wordpress — "Wordpress auto-updated for the first time despite auto-update not being enabled?" themes=['technical_cue']
- **FP_TPS** r/firefox — "Anybody else get the Firefox exe directly on the desktop after the 145 update? Normally it is a shortcut." themes=['technical_cue']
- **FP_TPS** r/MicrosoftEdge — "Most recent update added this intrusive toggle on my New Tab page. How can I get rid of it?" themes=['technical_cue']
- **FP_TPS** r/google — "I think… someone maybe pushed a bad update" themes=['technical_cue']
- **FP_TPS** r/comfyui — "ComfyUI 0.3.63: Subgraph Publishing, Selection Toolbox Redesign" themes=['uncategorized']
- **FP_TPS** r/chrome — "If You've Got TubeBlock, remove it!" themes=['short_title']
- **FP_TPS** r/comfyui — "Qwen edit images looking like this since updating Comfy" themes=['uncategorized']
- **FP_TPS** r/chrome — "Is this legit??Chrome Update" themes=['technical_cue', 'short_title']
- **FP_TPS** r/Wordpress — "Upgrading to PHP 8.3 causing pages to disappear?" themes=['uncategorized']
- **FP_TPS** r/rustdesk — "Updating ID" themes=['short_title']
- **FN_TPS** r/Android — "Samsung green lines issue is back! Users claim issue started after update" themes=['technical_cue']
- **FN_TPS** r/transformers — "Missing Link Ultra Magnus might be one of my favorite toys this year, especially after constantly disappointing updates" themes=['uncategorized']
- **FN_TPS** r/windows — "Microsoft wants to fix app updates on Windows 11 — previews new update orchestrator platform designed make them invisibl" themes=['technical_cue']
- **FN_TPS** r/linux — "Ubuntu 25.10 Unattended Upgrades Broken Due To Rust Coreutils Bug" themes=['technical_cue']
- **FN_TPS** r/comfyui — "Quick Update, Fixed the chin issue, Instructions are given in the description" themes=['technical_cue']
- **FN_TPS** r/Bitcoin — "Can't be unseen" themes=['short_title']
- **FN_TPS** r/angular — "I still can't get used to it 😀" themes=['short_title']
- **FN_TPS** r/Bitcoin — "Can't argue" themes=['short_title']