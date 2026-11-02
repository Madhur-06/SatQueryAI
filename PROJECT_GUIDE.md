# SatQuery AI: A Plain-Language Guide

This guide is for someone who knows only a little Python. It explains what this project
does, how the pieces fit together and how to run it. Technical words are explained the
first time they appear, and short forms are always spelled out.

> Everything below was checked against the real code on 29 September 2026. All 224
> automatic tests passed, and the example answers in Part 4 are real outputs from the
> program.

---

## Part 1: What is this project?

**In one sentence:** you give the program one or two satellite pictures and ask a
question in plain English. It works out the answer, shows you *where* in the picture
the answer comes from, tells you how sure it is, and keeps a full record of every
step it took.

It is being built for the **Smart India Hackathon (SIH) 2026**, problem statement
SIH26167, set by the **Indian Space Research Organisation (ISRO)**.

### Example questions it can answer

| You upload | You ask | It answers (roughly) |
|---|---|---|
| One satellite photo | "How much area is covered by water?" | "Water covers 16.68 hectares (10.2% of the scene)." |
| Two photos of the same place, taken years apart | "Has built-up area increased?" | "Yes. It went from 0 to 12.16 hectares between 2020 and 2024." |
| One normal photo + one radar image of the same place | "Map the water using optical and SAR" | Water area, plus how much the two sensors agree |

### A simple way to picture it: a hospital

Think of the program as a small hospital:

1. **Reception** checks your paperwork: are these real satellite files, and do the two
   pictures actually show the same place?
2. **The general doctor** reads your question and decides what kind of problem it is
   ("this is about water" or "this is about change over time").
3. **The general doctor writes a plan**: which specialists to send you to, and in what
   order.
4. **The specialists** (called *tools*) each do one job: one finds water, one measures
   area, one spots changes.
5. **The report writer** turns the specialists' results into one clear answer with a
   "how sure are we" score.
6. **Medical records** (the *trace*) store every step, so anyone can check the work later.

The code is organised in exactly these parts.

---

## Part 2: Words you will see (with full forms)

You don't need to memorise these. Come back here when you meet a word you don't know.

### About satellite images

| Word | Full form / meaning |
|---|---|
| **Optical image** | A normal "camera-like" satellite photo that uses sunlight. It can't see through clouds or at night. |
| **Multispectral image** | An optical image that records extra kinds of light that human eyes can't see (it has 5 or more "bands"). |
| **Band** | One layer of a satellite image. A normal photo has 3 bands (red, green, blue). Satellites often record more. |
| **RGB** | **R**ed, **G**reen, **B**lue: the three colours of normal light. |
| **NIR** | **N**ear-**I**nfra**r**ed. Invisible light just beyond red. Plants reflect a lot of it; water absorbs it. |
| **SWIR** | **S**hort-**W**ave **I**nfra**r**ed. Invisible light even further beyond red. Useful for spotting buildings and concrete. |
| **SAR** | **S**ynthetic **A**perture **R**adar. The satellite sends out radio waves and records the echo. It works at night and through clouds. Calm water looks very dark in it. |
| **VV / VH / HH / HV** | The "polarisation" of a radar image: how the radio wave was sent and received. **V** = **V**ertical, **H** = **H**orizontal. VV means "sent vertical, received vertical". |
| **dB** | **D**eci**b**el. A scale for measuring how strong the radar echo is. |
| **Speckle** | Grainy "salt and pepper" noise that every radar image has. |
| **Bi-temporal pair** | Two pictures of the same place taken on two different dates ("bi" = two, "temporal" = time). |
| **Co-registered** | Two images lined up so the same pixel shows the same spot on the ground. |
| **Pixel** | One tiny square of an image. In the demo images, each pixel is 10 m × 10 m on the ground. |
| **Hectare (ha)** | A unit of area: 100 m × 100 m = 10,000 m². About 2.5 acres. |
| **Mask** | A black-and-white picture that says "yes" or "no" for every pixel, for example "is this pixel water?". |

### About file formats

| Word | Full form / meaning |
|---|---|
| **TIFF** | **T**agged **I**mage **F**ile **F**ormat. An image file type. |
| **GeoTIFF** | **Geo**graphic TIFF. A TIFF that also stores *where on Earth* the picture is. This is the main input format. |
| **PNG** | **P**ortable **N**etwork **G**raphics. A normal picture file (no location stored). |
| **JPEG** | **J**oint **P**hotographic **E**xperts **G**roup. Another normal picture file (no location stored). |
| **JSON** | **J**ava**S**cript **O**bject **N**otation. A plain-text way to store data, like a Python dictionary. |
| **GeoJSON** | JSON that stores shapes on a map (for example, the outline of a lake). |
| **CRS** | **C**oordinate **R**eference **S**ystem. The "map projection" that says how the picture's pixels match real places on Earth. |
| **EPSG** | **E**uropean **P**etroleum **S**urvey **G**roup. They publish numbered codes for each CRS, such as "EPSG:32643". |

### About the calculations ("indices")

An *index* is a simple formula that takes two bands and gives one number per pixel.
High or low values point to a certain kind of land.

| Word | Full form | What it finds |
|---|---|---|
| **NDWI** | **N**ormalized **D**ifference **W**ater **I**ndex | Water |
| **MNDWI** | **M**odified **N**ormalized **D**ifference **W**ater **I**ndex | Water (works better near cities; needs SWIR) |
| **NDBI** | **N**ormalized **D**ifference **B**uilt-up **I**ndex | Buildings, roads, concrete |
| **NDVI** | **N**ormalized **D**ifference **V**egetation **I**ndex | Plants, trees, crops |
| **CVA** | **C**hange **V**ector **A**nalysis | Any change between two dates in optical images |
| **Log-ratio** | Comparing radar echo strength between two dates | Any change between two dates in radar images |
| **Otsu's method** | Named after Nobuyuki Otsu, who invented it | Automatically picks the best "cut-off" value to split pixels into two groups (for example water / not water) |
| **Lee filter** | Named after Jong-Sen Lee, who invented it | Smooths away radar speckle |
| **IoU** | **I**ntersection **o**ver **U**nion | A 0-to-1 score for how much two masks overlap. 1 means identical, 0 means no overlap. |

### About artificial intelligence and software

| Word | Full form / meaning |
|---|---|
| **VQA** | **V**isual **Q**uestion **A**nswering. Answering a question about a picture. |
| **VLM** | **V**ision-**L**anguage **M**odel. An artificial intelligence (AI) model that can look at pictures and write text. |
| **Qwen-VL** | A free VLM made by the company Alibaba ("VL" = **V**ision-**L**anguage). This project can optionally use "Qwen3-VL-2B" (2B = about 2 **b**illion internal numbers). |
| **Zero-shot** | Using an AI model as it is, without extra training for this specific job. |
| **Captioning** | Writing a short description of a picture. |
| **Grounding** | Pointing to *where* in the picture something is. |
| **Mock tool** | A **placeholder**. It has the right shape but gives a fake, fixed answer (always marked `[MOCK]`). It stands in the spot where a real AI model will go later. |
| **API** | **A**pplication **P**rogramming **I**nterface. A way for programs to talk to each other. Here, the website sends your pictures to the Python program through the API. |
| **UI** | **U**ser **I**nterface. The web page you click on. |
| **HTTP** | **H**yper**t**ext **T**ransfer **P**rotocol. The language web browsers and web servers use to talk. |
| **URL** | **U**niform **R**esource **L**ocator. A web address. |
| **CPU** | **C**entral **P**rocessing **U**nit. The computer's main processor. |
| **GPU** | **G**raphics **P**rocessing **U**nit. The graphics card. It runs AI models much faster than the CPU. |
| **CUDA** | **C**ompute **U**nified **D**evice **A**rchitecture. NVIDIA's software that lets programs use the graphics card. |

---

## Part 3: The journey of one question, step by step

Here is what happens when you upload a picture and ask **"How much area is covered by water?"**

```
 You (web page)
    │  picture file(s) + question
    ▼
 ① API / web server ........... src/satquery/api/app.py
    │  saves your files in data/requests/<a random id>/inputs
    ▼
 ② Input checking ............. src/satquery/io/validate.py
    │  "Is this a real satellite file? Optical or radar? Which band is which?
    │   If two files: do they show the same place?"
    ▼
 ③ Question classifier ........ src/satquery/agent/classifier.py
    │  reads the words: "water" → target is water; "how much" → wants an area
    ▼
 ④ Planner .................... src/satquery/agent/planner.py
    │  plan: 1) make a water mask  2) measure its area
    ▼
 ⑤ Executor ................... src/satquery/agent/executor.py
    │  runs each tool in order and catches any errors
    ▼
 ⑥ Tools (the specialists) .... src/satquery/tools/...
    │  ndwi_water_mask → finds water pixels
    │  area_stats      → counts them and converts to hectares
    ▼
 ⑦ Answer composer ............ src/satquery/agent/composer.py
    │  "Water covers 16.68 ha (10.2% of the scene) in 1 region."  confidence 0.95
    ▼
 ⑧ Evidence files + trace ..... src/satquery/agent/artifacts.py, trace.py
    │  pictures with the water coloured in, map files, and trace.json
    ▼
 Back to you (web page shows answer, pictures and the step-by-step record)
```

The single Python function that runs steps ② to ⑧ is `analyze()` in
`src/satquery/agent/pipeline.py`. The website, the tests and the benchmark scorer all
call this same function.

### Step ② in more detail: input checking

Before doing any work, the program makes sure the input makes sense:

- **File type.** Only GeoTIFF is allowed. PNG and JPEG are allowed only if you tick the
  "benchmark" option, because public test datasets use them.
- **What kind of image is it?** Optical, multispectral or radar. The program guesses
  from clues in the file: band names first, then the satellite name, then the pixel
  values. You can also tell it yourself.
- **Which band is which?** For example, "band 4 is near-infrared". It reads the band
  names inside the file. If they are missing, you can supply a *band map*, for example
  `{"green": 2, "nir": 4}`. It never guesses blindly from band order.
- **The date** each picture was taken. It reads this from inside the file, or from the
  file name (such as `s2_20200301.tif`).
- **For two pictures:**
  - Do they both have location information?
  - Do they overlap enough? By default, at least 50% overlap is required.
  - If they use different map projections, the second one is converted so its pixels line
    up with the first.
  - Two-date pairs are sorted so the older picture comes first.

After checking, the program sorts the input into one of **four input types**:

| Input type | Meaning |
|---|---|
| `single_opt` | One optical picture |
| `single_sar` | One radar picture |
| `pair_opt_sar` | One optical + one radar picture of the same place |
| `pair_temporal` | Two pictures of the same place on two dates |

If something is wrong, it stops and gives a **clear error with a code and a hint**.
For example, `no_overlap` means "your two pictures don't show the same place". The full
list of error codes is in `src/satquery/errors.py`.

### Step ③ in more detail: understanding the question

This is done by **simple keyword matching**, not by AI (file `classifier.py`). It looks for:

- **What thing?** Words like *water, river, lake, flood* mean water. *Building, urban,
  city* mean built-up area. *Forest, tree, crop* mean vegetation.
- **What kind of answer?**
  - A question starting with *is / are / has / does* wants **yes or no**.
  - *How many* wants a **count**.
  - *How much / area / percent* wants an **area**.
  - Anything else wants **free text**.
- **What kind of job?**

  | Job | Chosen when |
  |---|---|
  | Change question | Words like *increase, decrease, changed, before, after*, or you uploaded two dates |
  | Caption | Words like *describe, summarise* |
  | Grounding | Words like *where, locate, show me* |
  | Fusion | Words like *optical and SAR, combine*, or you uploaded optical + radar and named a thing |
  | Normal question (VQA) | Everything else |

It also writes down *why* it decided each thing. You can see these reasons in the trace.

### Step ④ in more detail: the planner

The planner follows a few **recipes** (fixed step lists):

| Recipe | When | Steps |
|---|---|---|
| **mapping** | One picture, a named thing | make mask → measure area |
| **fusion** | Optical + radar | optical mask + radar mask → combine them → measure area |
| **class_change** | Two dates, a named thing | mask on date 1 + mask on date 2 → compare (gained / lost) → measure → double-check with CVA |
| **generic_change** | Two dates, no named thing (or it can't be mapped) | detect any change → measure |
| **vlm** | Describe / general questions | ask the language tool (Qwen AI model, or the mock placeholder) |

**Clever part:** a recipe never names a specific tool. It only says what it needs, for
example "I need a *water mask*". Then it asks the **registry** (the list of all tools,
in `tools/registry.py`) for the best tool that can do that job on *these* pictures. The
registry ranks tools like this:

1. Is it switched on and working?
2. Is it real, not a mock?
3. Which has the higher "priority" number?

So when someone adds a better water-finding AI model later, the planner will use it
automatically, with no other code changes.

**Fallback.** If the question names something the tools can't map on this picture (for
example "water" on a plain RGB photo, which has no near-infrared band), yes/no and
count questions are handed to the language tool instead, with a warning. Area questions
are refused with a helpful error, because only a mask can give hectares.

### Step ⑤: the executor

The executor runs each planned step in order. If one step crashes, only that step fails.
Any step that needed its result is marked "skipped", and the program still returns
whatever it can.

### Step ⑦: the confidence score (how sure it is)

Confidence is a number from 0 to 1. It is worked out with simple, openly written rules
(the full rules are at the top of `composer.py`):

- **Mask tools:** confidence is high when most pixels are far from the cut-off value
  (clearly water or clearly not water), and low when many are borderline.
- **Fusion:** confidence goes up when the optical and radar results agree.
- **Change:** confidence goes up when the independent CVA change check agrees with the
  result.
- **Rule-based answers** are **never allowed above 0.95**, because a hand-made rule is
  never certain.
- **Mock (placeholder) answers** always get **0**.
- **Qwen AI answers** get a fixed **0.5**, because the model doesn't report how sure it is.

### Step ⑧: what you get back

Inside `data/requests/<id>/outputs/` you will find:

| File | What it is |
|---|---|
| `quicklook_*.png` | A normal-looking preview of each input picture |
| `*_overlay.png` | The preview with the answer area coloured in (for example water in blue) |
| `*_mask.png` | Just the coloured area on a see-through background |
| `*.tif` | The mask as a GeoTIFF, which keeps its real-world location and opens in mapping software such as QGIS |
| `*.geojson` | The outline of the area as map shapes |
| `trace.json` | **The full record**: inputs, what the question was understood as, the plan and why, every step's settings, time taken, results and warnings |

---

## Part 4: Real example outputs

These came from running the program on the built-in demo pictures:

| Question | Input | Tools used, in order | Answer | Confidence |
|---|---|---|---|---|
| How much area is covered by water? | 1 optical | `ndwi_water_mask` → `area_stats` | Water covers 16.68 ha (10.2% of the scene) in 1 region. | 0.95 |
| Has built-up area increased? | 2 dates | `ndbi_builtup_mask` ×2 → `mask_compare` → `area_stats` → `change_cva` | Yes. Built-up area increased between 2020-03-01 and 2024-03-01: from 0.00 ha to 12.16 ha (gained 12.16 ha, lost 0.00 ha, 1 new region). An independent spectral change detector (`change_cva`) agrees on 100% of the changed area. | 0.95 |
| Map the water using optical and SAR | optical + radar | `ndwi_water_mask` + `sar_water_otsu` → `optsar_fusion_rules` → `area_stats` | Water covers 16.68 ha. Optical and SAR agree on 16.67 ha (100%). | 0.95 |
| Describe this image | 1 optical | `vlm_caption` (mock) | `[MOCK]` An optical image, 128x128 px, pixel size 10. | 0.0 |

The last row shows a **mock**: describing a picture in words needs the AI model, which
is switched off by default.

---

## Part 5: The specialist tools

All tools live in `src/satquery/tools/`. Every tool follows the same template, defined
in `tools/base.py`. Each one states its name, what job it does, what images and bands
it needs, and which settings are allowed.

### Real tools (plain maths, no AI), in `tools/classical/`

| Tool name | File | What it does in plain words |
|---|---|---|
| `ndwi_water_mask` | `indices.py` | Finds water in optical pictures using green and near-infrared light (or SWIR if available). |
| `ndbi_builtup_mask` | `indices.py` | Finds buildings and concrete using SWIR and near-infrared light. Plants and water are removed first. **Needs a SWIR band**, so a plain 4-band picture can't use it. |
| `ndvi_vegetation_mask` | `indices.py` | Finds plants using near-infrared and red light. |
| `sar_water_otsu` | `sar_water.py` | Finds water in radar pictures: smooths the speckle, then marks very dark pixels as water. It warns you if the "water" isn't dark enough to be real water. |
| `optsar_fusion_rules` | `fusion.py` | Combines an optical mask and a radar mask. It reports where both agree, where only optical sees it, and where only radar sees it. |
| `change_cva` | `change.py` | Finds anything that changed between two optical pictures. |
| `sar_logratio_change` | `change.py` | Finds anything that changed between two radar pictures, and whether the echo got stronger or weaker. |
| `mask_compare` | `change.py` | Compares one thing (such as buildings) on date 1 against date 2: what was gained and what was lost. |
| `area_stats` | `area.py` | Measures any mask: number of pixels, hectares, percentage of the picture, and number of separate patches. |

Small helper maths shared by these tools (cut-off picking, cleaning tiny specks,
confidence) is in `tools/classical/_common.py`.

### Placeholder tools, in `tools/mock/__init__.py`

`vlm_vqa`, `vlm_caption`, `grounding` and `change_vqa` give fixed answers: "yes" for
yes/no questions, "0" for counts. They are always marked **[MOCK]** with confidence 0.
They exist so the whole system works end to end today, and real AI models can slot into
their place later.

### The optional AI tool, in `tools/ml/qwen_vl.py`

`qwen_vl_zeroshot` uses the **Qwen3-VL-2B** AI model to answer questions about pictures,
describe them, and compare two dates. It is **off by default**. When switched on, it
automatically wins over the mocks because of its higher priority. Part 7 explains how to
switch it on.

---

## Part 6: Folder and file tour

```
SAT_QueryAI/
├── README.md            ← technical overview (for programmers)
├── PROGRESS.md          ← diary: what's done, what's next, open questions
├── PROJECT_GUIDE.md     ← this file
├── CLAUDE.md            ← project rules for the AI coding assistant
├── pyproject.toml       ← list of Python packages the project needs
├── uv.lock              ← exact versions of those packages (don't edit by hand)
├── justfile             ← shortcut commands (just install, just serve, ...)
│
├── src/satquery/        ← ALL THE PYTHON CODE
│   ├── types.py         ← shared names: input types, job types, band names
│   ├── config.py        ← settings (folders, upload limit, AI on/off)
│   ├── errors.py        ← the list of error codes and the error format
│   │
│   ├── io/              ← "Reception": reading and checking pictures
│   │   ├── raster.py        reads an image file and its details
│   │   ├── modality.py      guesses optical vs radar
│   │   ├── bands.py         works out which band is red, NIR, SWIR, VV...
│   │   ├── dates.py         finds the date the picture was taken
│   │   ├── validate.py      main checker; builds the "input bundle"
│   │   ├── align.py         lines up a second picture onto the first
│   │   ├── geometry.py      overlap and area-on-Earth maths
│   │   ├── georaster.py     a picture array that remembers its map location
│   │   ├── render.py        makes preview images and coloured overlays
│   │   ├── vector.py        turns masks into map outlines (GeoJSON)
│   │   ├── tiling.py        cuts big pictures into pieces (ready, not used yet)
│   │   └── preprocess/      cleaning steps: radar smoothing, brightness
│   │                        stretch, matching two dates
│   │
│   ├── agent/           ← "Doctor + report writer": the decision-making brain
│   │   ├── pipeline.py      analyze(): runs everything start to finish
│   │   ├── classifier.py    understands the question (keywords)
│   │   ├── planner.py       picks a recipe and builds the plan
│   │   ├── plan.py          what a plan and a plan step look like
│   │   ├── executor.py      runs the plan's steps safely
│   │   ├── composer.py      writes the final answer + confidence
│   │   ├── artifacts.py     saves the evidence pictures and map files
│   │   └── trace.py         the step-by-step record (trace.json)
│   │
│   ├── tools/           ← "Specialists" (see Part 5)
│   │   ├── base.py          the template every tool follows
│   │   ├── registry.py      the list of tools + "find the best tool for this job"
│   │   ├── classical/       real maths-based tools
│   │   ├── mock/            placeholder tools
│   │   └── ml/              the optional Qwen AI tool
│   │
│   ├── api/             ← the web server
│   │   ├── app.py           web addresses: /analyze, /tools, /health, files
│   │   ├── schemas.py       the shape of the replies
│   │   └── storage.py       saves uploads safely in data/requests/
│   │
│   ├── eval/            ← "Report card": scoring the system on public datasets
│   │   ├── cli.py           the satquery-eval command
│   │   ├── runner.py        runs a tool on many questions and scores it
│   │   ├── download.py      downloads datasets (only when you ask)
│   │   ├── sources.py       where each dataset comes from, and its licence
│   │   ├── report.py        writes results into results/
│   │   ├── metrics/         how answers are scored (accuracy, overlap...)
│   │   └── datasets/        readers for each dataset  (⚠ see Part 10)
│   │
│   └── testing/         ← makes fake satellite pictures for tests and demos
│       ├── synthetic.py     draws fake pictures with a known lake and buildings
│       ├── demo.py          writes demo pictures to data/demo
│       └── helpers.py       small test helpers
│
├── tests/               ← 224 automatic checks that the code works
├── ui/                  ← the web page (written in TypeScript + React)
│   └── src/
│       ├── App.tsx              the main page
│       ├── api.ts               sends your upload to the Python server
│       └── components/
│           ├── UploadForm.tsx     file picker + question box
│           ├── AnswerCard.tsx     shows the answer, confidence, errors
│           ├── OverlayViewer.tsx  shows pictures with coloured layers
│           └── TraceViewer.tsx    shows the step-by-step record
│
├── results/             ← saved benchmark scores (see Part 8)
├── data/                ← your uploads, outputs and demo pictures (not saved in git)
├── datasets/            ← downloaded benchmark datasets (not saved in git)
└── venv/                ← OLD, unused Python environment; safe to delete
```

**About the testing pictures:** the tests never use real satellite data. `synthetic.py`
*draws* fake pictures: a 128×128-pixel scene with a round lake, some fields and, in the
"2024" picture, a new block of buildings. Because the program knows exactly where the
fake lake is, the tests can check that the answer is correct.

---

## Part 7: How to run it (Windows)

Open **PowerShell** or the terminal in your code editor, and go to the project folder:

```powershell
cd C:\Users\Madhur\Desktop\SAT_QueryAI
```

### What you need installed

All of these are **already installed on this computer**:

| Program | What it is for | Check with |
|---|---|---|
| Python 3.12 | Runs the code | `python --version` |
| uv | Installs Python packages and runs the project | `uv --version` |
| just | Runs the shortcut commands in `justfile` | `just --version` |
| Node.js 20 or newer | Builds the web page | `node --version` |

To see every shortcut command, type:

```powershell
just
```

### First-time setup (skip this on this computer: it's already done)

On a **fresh** computer you would run:

```powershell
just install
```

This creates a private Python environment in `.venv`, installs all packages and installs
the web page's packages.

> ⚠ **On this computer, do not run `just install` again.** The GPU (graphics card)
> version of the AI packages is installed here. `just install` would *remove* them.
> If you ever need to reinstall, use `just install-gpu` instead.

### Option A: use the web page (easiest)

**1. Make the demo pictures** (only needed once):

```powershell
just demo-data
```

This writes fake satellite pictures into `data\demo\`.

**2. Start the program:**

```powershell
just serve
```

Wait until you see a line like `Uvicorn running on http://127.0.0.1:8000`.

**3. Open your web browser** and go to **http://127.0.0.1:8000**.

**4. Try it:**
- Click the file picker and choose `data\demo\optical_4band.tif`.
- Pick the question "How much area is covered by water?".
- Click the button to analyse.
- You'll see the answer, the confidence, the picture with water coloured in, and the
  step-by-step trace.

**5. Try the other demos:**

| Pick these files | Ask |
|---|---|
| `data\demo\sar_vv_vh.tif` | Is there any water in this image? |
| `data\demo\optical_4band.tif` **and** `data\demo\sar_vv_vh.tif` (select both) | Map the water using optical and SAR |
| `data\demo\temporal\s2_20200301.tif` **and** `data\demo\temporal\s2_20240301.tif` | Has built-up area increased? |

**6. To stop the program,** click in the terminal and press **Ctrl + C**.

### Option B: use it from Python directly

Create a small file, for example `try_it.py`, in the project folder:

```python
from pathlib import Path
from satquery.agent.pipeline import analyze

result = analyze(
    [Path("data/demo/optical_4band.tif")],
    "How much area is covered by water?",
    out_dir=Path("data/my_test"),  # where the pictures and trace.json are saved
)
print("Status:", result.trace.status)
print("Answer:", result.answer.text)
print("Confidence:", result.answer.confidence)
```

Run it with:

```powershell
uv run python try_it.py
```

`uv run` means "run this inside the project's private Python environment". Always put
`uv run` in front of Python commands for this project.

### Run the automatic tests

```powershell
just test
```

You should see something like `224 passed, 1 skipped`. The skipped test is the one that
loads the real AI model. Run that one with `just test-ml`.

### Switch on the Qwen AI model (optional)

The AI packages and the model files (about 4 GB) are **already on this computer**, and
your NVIDIA graphics card is detected. To use the model:

1. Create a file named `.env` in the project folder containing this one line:
   ```
   SATQUERY_VLM_ENABLED=true
   ```
2. Restart `just serve`.
3. Now "Describe this image" and general questions get real AI answers instead of
   `[MOCK]`.

The first question takes a little longer while the model loads. After that, it takes
about half a second per question on the graphics card (about 6 seconds on the CPU).

**Other settings** can go in the same `.env` file:

| Setting | Default | Meaning |
|---|---|---|
| `SATQUERY_DATA_DIR` | `data` | Folder for uploads and outputs |
| `SATQUERY_MAX_UPLOAD_MB` | `512` | Biggest allowed upload, in megabytes |
| `SATQUERY_MIN_OVERLAP_IOU` | `0.5` | How much two pictures must overlap (0 to 1) |
| `SATQUERY_VLM_ENABLED` | `false` | Switch the Qwen AI model on or off |
| `SATQUERY_VLM_MODEL_ID` | `Qwen/Qwen3-VL-2B-Instruct` | Which AI model to use |
| `SATQUERY_VLM_ALLOW_DOWNLOAD` | `false` | Allow downloading the model from the internet |

### For programmers: live-reload mode

Use this while editing the code. Open **two** terminals:

```powershell
just dev-api     # terminal 1: Python server, restarts when you save a .py file
just dev-ui      # terminal 2: web page, refreshes when you save a .tsx file
```

Then open **http://localhost:5173**.

---

## Part 8: The "report card" (benchmarks)

To show how good the system is, it is tested on **public datasets**: large collections
of satellite pictures with known correct answers.

| Dataset | Full name | What's in it |
|---|---|---|
| RSVQA-LR | **R**emote **S**ensing **V**isual **Q**uestion **A**nswering, **L**ow **R**esolution | About 10,000 questions about satellite pictures |
| CDVQA | **C**hange **D**etection **V**isual **Q**uestion **A**nswering | About 40,000 questions about changes between two dates |
| VRSBench | **V**ersatile **R**emote **S**ensing **Bench**mark | Captions, questions and "where is it" tasks (reader ready, not downloaded yet) |
| BigEarthNet.txt | BigEarthNet with text | Multispectral pictures with text (reader ready, not tested on real data) |

RSVQA-LR and CDVQA are already downloaded into `datasets\`.

**Useful commands:**

```powershell
uv run satquery-eval --list                                             # list datasets and tools
uv run satquery-eval --benchmark rsvqa-lr --tool pipeline --subset 100  # score 100 random questions
```

Results are saved in `results\`. `results\README.md` has the summary table.

**Score words:**
- **OA** = **O**verall **A**ccuracy: the percentage of all questions answered correctly.
- **AA** = **A**verage **A**ccuracy: the accuracy for each question type, averaged, so rare
  types count equally.

**Scores so far (percent correct):**

| Dataset | Placeholder (always "yes" / "0") | Full pipeline without AI | Qwen AI model |
|---|---|---|---|
| RSVQA-LR (500 questions) | 42.8 | 42.8 | **47.8** |
| CDVQA (1,000 questions) | 29.2 | 29.2 | **42.5** |

Why is the "full pipeline without AI" no better than the placeholder? These public
datasets use normal RGB photos. They have no near-infrared or SWIR light, so the
maths-based tools can't work on them, and the pipeline falls back to the placeholder.
The maths tools are built for real multispectral satellite data, and on the demo data
they are very accurate (over 90% overlap with the true answer).

---

## Part 9: What works today, and what doesn't yet

### Works ✅

- All four input types, from upload to answer, with evidence pictures and a full trace
- Finding water, buildings and vegetation, with areas in hectares
- Change over time, with gained and lost areas, and a double-check
- Optical + radar combined, with an "agreement" score
- Clear error messages with codes and hints for bad inputs
- Web page, API and 224 passing tests
- Optional Qwen AI model (works on this computer's graphics card)
- Benchmark scoring on two real public datasets

### Not yet / known limits ⚠

- **The question reader is keyword-based.** Unusual wording can be misunderstood. The
  plan is to replace it with an AI model later; the code is already set up for that swap.
- **Only one thing per question.** "Water *and* buildings" analyses only the first one
  (with a warning).
- **Building detection needs a SWIR band.** A plain 4-band picture gets a clear
  `band_role_missing` error.
- **Change detection always finds *some* change.** The automatic cut-off always splits
  pixels into two groups. So on the CDVQA dataset it always says "yes".
- **The Qwen AI only answers with one word or a number.** Its instructions ask for
  "yes / no / a number", so questions whose answer is a word (such as "rural or urban?")
  always score 0. This is planned to be fixed.
- **Captioning and grounding** are placeholders unless the Qwen AI is switched on.
  Grounding has no real tool yet.
- **Whole pictures are loaded into memory.** Very large satellite scenes could be slow.
  The code to cut them into tiles exists but isn't connected yet.
- **The web page hasn't been tested in a real browser yet.** It has been checked through
  the API only.

The "Next" and "Open questions" sections of `PROGRESS.md` have the full list.

---

## Part 10: Things to know

1. **⚠ Some code is not saved in git.** The file `.gitignore` tells git which files to
   ignore. It has the line `datasets/`, meant for the big downloaded dataset folder. By
   accident, that line *also* matches the code folder `src/satquery/eval/datasets/`, so
   that code has **never been saved in git**. If someone copies the project from git,
   the `satquery-eval` command will break. The fix is to change the line to `/datasets/`
   (the leading slash means "only the top-level folder") and then save the folder in git.
2. **The warning `VIRTUAL_ENV=venv does not match...`** is harmless. It appears because
   the old, unused `venv` folder is still switched on in your terminal. You can type
   `deactivate` to switch it off, or delete the old `venv` folder.
3. **The "benchmark" tick box** allows PNG/JPEG files. Normal photos have no location
   information, so areas come out in *pixels* instead of hectares.
4. **Nothing is downloaded automatically.** Datasets and AI models are downloaded only
   when you run a download command yourself.

---

## Part 11: Common problems

| Problem | What to do |
|---|---|
| `just: command not found` | Install it with `uv tool install rust-just`, or use the plain commands in the next table. |
| The web page says "Is the API running?" | Start the program with `just serve` and keep that terminal open. |
| Opening http://127.0.0.1:8000 shows `{"detail":"Not Found"}` instead of the page | The web page wasn't built. Run `just ui-build`, then `just serve`. |
| Error `unsupported_format` | Upload a GeoTIFF (`.tif`), or tick "benchmark" for PNG/JPEG. |
| Error `band_role_missing` | The picture lacks a band the tool needs (such as SWIR for buildings). If the band exists but has no name, fill in the band map, for example `{"nir": 4, "swir1": 5}`. |
| Error `no_overlap` / `low_overlap` | The two pictures don't show the same place. |
| Error `task_input_mismatch` | The question doesn't fit the input. For example, a "change" question needs two dates. |
| The answer says `[MOCK]` | That job needs the AI model. Switch it on (see Part 7). |
| Port 8000 is already in use | Another copy is still running. Close it, or run `just --set port 8001 serve`. |

### Commands without `just`

If `just` isn't available, these do the same things:

| `just` command | Plain command |
|---|---|
| `just demo-data` | `uv run python -m satquery.testing.demo data/demo` |
| `just test` | `uv run pytest` |
| `just ui-build` | `npm --prefix ui run build` |
| `just serve` | build the page first (line above), then `uv run uvicorn satquery.api.app:app --host 127.0.0.1 --port 8000` |
