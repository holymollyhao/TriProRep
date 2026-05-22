# DNAChunker Demo

Static project page + interactive chunking visualizer for the released
DNAChunker checkpoint. The page lives on GitHub Pages; the inference
backend is the FastAPI app under `server/`, which loads the checkpoint
(`../code_release/pretrained_ckpt/last.ckpt`) and runs a forward pass on GPU.

## Layout

```
demo_release/                 # GitHub Pages root
├── index.html                # served at /
├── static/                   # served at /static
│   ├── app.js
│   ├── style.css
│   ├── *.png
│   └── flama_font/
├── server/                   # NOT served by Pages — local/private backend
│   ├── app.py
│   ├── inference.py
│   └── requirements.txt
├── .nojekyll                 # disables Jekyll on Pages
├── .gitignore
└── README.md
```

## Deploying to GitHub Pages (frontend only)

The repo is intended to be the root of a private GitHub repo with Pages enabled.

1. `cd demo_release && git init && git remote add origin git@github.com:<user>/<repo>.git`
2. Commit and push to `main`.
3. On GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a
   branch → Branch: `main` / `/ (root)`**.
4. After ~1 min the site is live at `https://<user>.github.io/<repo>/`.

`index.html` uses relative paths, so it works under any sub-path Pages assigns.
`.nojekyll` keeps Pages from filtering underscore-prefixed files.

## Wiring the live demo to a backend

In production the static page on `*.github.io` cannot reach `localhost`, so
the FastAPI server must be exposed at a public HTTPS URL. The frontend reads
the URL from `static/app.js`:

```js
const API_BASE =
  window.location.hostname.endsWith("github.io")
    ? "https://dnachunker.CHANGE-ME.example.com"   // <-- replace
    : "";
```

Replace the placeholder with the URL of your tunnel / cloud endpoint. The
recommended setup keeps the inference machine fully closed to inbound traffic:

- Run `server/app.py` on the GPU machine, bound to `127.0.0.1:8000`.
- Expose it with `cloudflared tunnel` (outbound only — no open ports).
- Add CORS allowlist for your Pages origin:
  `DNA_CHUNKER_CORS="https://<user>.github.io" uvicorn app:app`

The server already validates `sequence` server-side against `^[ACGTNacgtn]{1,8192}$`
and rejects anything else with 400.

## Local dev

```bash
conda activate tokenize
pip install -r server/requirements.txt
cd server
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open <http://localhost:8000>. The 2 GB checkpoint is loaded at startup, so the
first boot is slow; subsequent requests are fast.

### Env vars

| var | default | meaning |
| --- | --- | --- |
| `DNA_CHUNKER_CKPT`   | `../../code_release/pretrained_ckpt/last.ckpt` | checkpoint |
| `DNA_CHUNKER_CONFIG` | `../../code_release/configs/pretrain/default.yaml` | model config |
| `DNA_CHUNKER_CORS`   | localhost only | comma-separated CORS allowlist |

## API

`POST /api/chunk` — body `{ "sequence": "ACGT..." }`. Returns per-base Stage-1
and Stage-2 chunk IDs, boundary probabilities, and MLM predicted bases.

`GET /api/health` — quick readiness check (device, dtype, max length).
