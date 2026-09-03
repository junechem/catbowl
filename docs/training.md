# Teaching it your cats

The model is not trained from scratch. A frozen ImageNet backbone
(`mobilenet_v3_small`) turns each crop into a 576-number vector, and a logistic
regression on top learns to separate three cats from a few hundred vectors.
Training takes seconds and can be redone on the Pi itself.

## Which Python

Every `python -m catbowl ...` below assumes the project's virtual environment is
active. It holds torch, torchvision and scikit-learn; your system Python almost
certainly does not, and you will get `ModuleNotFoundError: No module named
'torch'` if you use it by mistake.

Either activate it for the session:

```
source .venv/bin/activate         # bash / zsh
source .venv/bin/activate.fish    # fish
```

or spell out the interpreter every time, which needs no activation:

```
.venv/bin/python -m catbowl train
```

No `.venv` yet? `scripts/install_pi.sh` makes one on the Pi; on a workstation:

```
python3 -m venv .venv && .venv/bin/pip install -r requirements-train.txt
```

## 0. Test the rig before you have a model

You do not need a trained model, or even a second cat, to check that the whole
chain works. Detection and identity are separate stages, and you can run the
first without the second:

```
python -m catbowl run --no-model
```

Every detection is treated as that bowl's own cat. Camera, motion detector,
vote window, confirmation timer, lid slew and close delay all run exactly as
they will in service - only "which cat is this" is stubbed out.

This is how you tune camera framing, `detector.min_area_frac`, and the policy
timings before collecting a single photo.

With the default `detector.type: hybrid` this opens **for any cat** - the gate
still checks that what arrived is a cat, it just does not ask which one. That
makes it a genuine test of the mechanism: if a lid lifts for you waving a hand
at the camera, the gate is misconfigured, not the model.

Set `detector.type: motion` if you want it to fire on anything at all, which is
easier to trigger while you are alone at the bench.

Either way, do not leave it running unattended with food in the bowls. Add
`--dry-run` to watch the decisions with the servos simulated and nothing moving.

## 1. Import the phone photos you already have

One folder per cat, then:

```
python -m catbowl import --src ~/Photos/mochi   --label mochi
python -m catbowl import --src ~/Photos/pepper  --label pepper
python -m catbowl import --src ~/Photos/biscuit --label biscuit
```

Each photo is run through an SSDLite COCO detector, the cat is cropped out, and
the crop lands in `data/crops/<label>/`. Photos where no cat is found are skipped
and counted; if most of your photos are already tight close-ups, re-run with
`--keep-uncropped`.

Aim for **60+ crops per cat**, varied across lighting, angle and how curled up
they are. The cats never need to be photographed at the same time, or even in
the same week - training reads one folder per cat and never sees them together.
Two cats' folders is the minimum: a classifier with one class has nothing to
separate, and `train` will refuse. Photos where two cats are in frame are worse than useless — the
detector picks one and may label the wrong animal. Set those aside.

## 2. Capture from the actual rig

Photos from your phone and frames from a fixed webcam under kitchen lighting do
not look alike, and the difference costs real accuracy. So once the cameras are
mounted, get each cat to the bowl and record:

```
python -m catbowl capture --bowl bowl1 --label mochi --seconds 120
```

This uses that bowl's camera and detector settings and saves a crop every 0.4 s
while something is moving. Hold still for the first few seconds — the motion
detector is learning the empty scene. Fifty on-rig crops per cat are worth more
than two hundred phone photos.

### The `_other` class

Not optional, really. A logistic regression over three cats has probabilities
that sum to one, so it *must* return one of them for anything it is shown - and
out-of-distribution inputs are exactly where it is confidently wrong, so
`min_confidence` alone will not save you.

`detector.type: hybrid` is the first line of defence: it establishes that a cat
is present before the classifier is asked which cat. `_other` is the second,
for the cats it does gate through that are not yours.


Make a `data/crops/_other/` folder with crops of anything that is *not* one of
your cats: a hand reaching in, a neighbour's cat at the window, the empty bowl,
a dog. `_other` is trained as a normal class but can never win a vote, so it
gives the model an explicit place to put "something is there, but not a cat I
know" instead of forcing it toward the nearest cat.

The neighbour's cat is the case that matters most, because it is the one the
detector gate will happily pass through.

## 3. Train

```
python -m catbowl train
```

Output:

```
images: 412  vectors: 824  raw accuracy: 97.1%

confusion matrix (rows = truth, columns = predicted)
              biscuit    mochi   pepper
   biscuit         34        1        0
     mochi          0       36        2
    pepper          1        0       29

confidence threshold sweep
  thresh  coverage  precision
    0.40     98.1%      96.2%
    0.50     96.1%      97.0%
    ...
suggested recognition.min_confidence: 0.72
```

Read the confusion matrix, not just the accuracy. If two cats are confused with
each other specifically, more photos of *those two* is the fix.

Put the suggested threshold into `config/bowls.yaml`. The trade-off it encodes:

- **Higher** → fewer wrong lids opening, more "the bowl ignored me" moments.
- **Lower** → the opposite.

For cats on different prescription diets, bias high. `--target-precision 0.995`
makes `train` suggest a stricter value.

## 4. Check it, then keep improving it

```
python -m catbowl eval --data data/crops --threshold 0.8
```

reports how many crops would open the right lid, the wrong lid, or no lid, and
lists the specific files it got wrong — look at those images, they usually
explain themselves (motion blur, half a cat, the tail end of a cat).

Then just let the feeder run. With `capture.dir` set — it is `data/collected` as
shipped — every detection banks a crop in `data/collected/unsorted/`, one every
`capture.interval_s`, up to `capture.max_images`. Filenames are
`<bowl>-<date>-<time>.jpg`, so the bowl tells you which cat is *likely* in the
frame, but nothing is labelled for you: before a classifier exists the rig has no
honest way to know, and folders named by a guess are worse than no folders.

After a week, file them: open `http://<pi>:8080/sort` on a phone and give each
photo a bucket (`capture.labels`, `J`/`K`/`F`/`M` as shipped, where M is
more-than-one-cat). They land in `data/collected/<label>/`, and photos worth
nothing go to `data/collected/discard/` - a folder, not a delete, because an
empty-bowl frame is exactly what a later model needs to learn to refuse.

Got one wrong? `http://<pi>:8080/browse` lists any bucket newest-first - a
mistake is nearly always one just made, so it is in the top-left corner - and a
tap moves it to another bucket or back into the sorting queue.

Then move the buckets you want to train on into `data/crops/<cat>/` and retrain. This is where most of the eventual
accuracy comes from.

`--collect` is the older, narrower version of the same idea: it saves only the
crop behind each state change, filed under whatever the classifier decided. It is
useful once a model exists and you want to audit its mistakes.

## If two of your cats look nearly identical

Vision alone struggles with, say, two black shorthairs. Options, in order:

1. More on-rig crops, especially of the confusable pair.
2. Raise `min_confidence` and `votes_required` — slower, but far fewer mistakes.
3. Switch `recognition.model` to `mobilenet_v3_large` (more accurate, ~3× slower
   on a Pi 4; still fine at `loop_fps: 3`) and retrain.
4. Add a distinguishing collar tag, or fall back to an RFID collar reader as the
   identity source. Not what you asked for, but it is what commercial selective
   feeders do, and it is honest about where pure vision runs out.
