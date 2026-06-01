# Error Analysis — roberta

**OOF source:** `evaluation/outputs/paper_roberta_5fold/roberta/oof_predictions.csv`
**Generated:** 2026-05-26T21:03:56.994284+00:00

- **OOF accuracy:** 79.3%
- **False positives (predicted TPS, gold GDS):** 57
- **False negatives (predicted GDS, gold TPS):** 55

## Failure themes (title heuristics)

### False positives — predicted TPS

- **uncategorized:** 28 posts
- **short_title:** 19 posts
- **technical_cue:** 11 posts
- **news_announcement:** 2 posts
- **rant_emotion:** 1 posts
- **crypto:** 1 posts

### False negatives — missed TPS

- **uncategorized:** 23 posts
- **short_title:** 16 posts
- **technical_cue:** 15 posts
- **news_announcement:** 2 posts
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
- **FP_TPS** r/Wordpress — "Wordpress auto-updated for the first time despite auto-update not being enabled?" themes=['technical_cue']
- **FP_TPS** r/firefox — "Anybody else get the Firefox exe directly on the desktop after the 145 update? Normally it is a shortcut." themes=['technical_cue']
- **FP_TPS** r/MicrosoftEdge — "Most recent update added this intrusive toggle on my New Tab page. How can I get rid of it?" themes=['technical_cue']
- **FP_TPS** r/ios — "Appstore unavailable after updated to ios 26.2.1" themes=['uncategorized']
- **FP_TPS** r/Android — "One UI 8.5: Calculator app gets an update to v12.5.10.18." themes=['technical_cue', 'news_announcement']
- **FP_TPS** r/chrome — "Is this legit??Chrome Update" themes=['technical_cue', 'short_title']
- **FP_TPS** r/Wordpress — "Upgrading to PHP 8.3 causing pages to disappear?" themes=['uncategorized']
- **FP_TPS** r/immich — "Migration from Old Storage Configuration" themes=['short_title']
- **FP_TPS** r/ansible — "Delegate_to: localhost gives me trouble" themes=['short_title']
- **FP_TPS** r/typescript — "@ts-ignore is almost always the worst option" themes=['rant_emotion']
- **FP_TPS** r/MicrosoftEdge — "My Edge randomly my search engine Yandex? I don't want to use yandex and didn't switch it myself" themes=['uncategorized']
- **FN_TPS** r/chrome — "My chrome randomly freezes every after a few hours ever since I upgraded my Mac(M1) to MacOS Tahoe" themes=['uncategorized']
- **FN_TPS** r/immich — "Immich_Server in Container Manager Stopped Unexpectedly constantly after updating to V2.0.0" themes=['uncategorized']
- **FN_TPS** r/Android — "Samsung green lines issue is back! Users claim issue started after update" themes=['technical_cue']
- **FN_TPS** r/immich — "Unable to upgrade to the latest version server" themes=['uncategorized']
- **FN_TPS** r/transformers — "Missing Link Ultra Magnus might be one of my favorite toys this year, especially after constantly disappointing updates" themes=['uncategorized']
- **FN_TPS** r/immich — "Updated to 1.10.6 and cannot see pictures anymore" themes=['uncategorized']
- **FN_TPS** r/windows — "Microsoft wants to fix app updates on Windows 11 — previews new update orchestrator platform designed make them invisibl" themes=['technical_cue']
- **FN_TPS** r/comfyui — "Quick Update, Fixed the chin issue, Instructions are given in the description" themes=['technical_cue']