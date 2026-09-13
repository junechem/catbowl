# The state of the catbowl

Where the rig is, what is wrong with it, and what happens next.
Updated 2026-09-13. Keep it updated: it is the only place most of this is
written down.

## Where it is now

The hardware works. One bowl is live (`bowl1`): a cardboard disc lying on the
bowl, turned aside by a single servo on PCA9685 channel 1, now glued to the lid
rather than taped. The camera is a Logitech Brio 100 on USB. The status page
serves the live view, the sorting queue and the browser at
`http://rjwpi.local:8080/`.

A classifier now exists, trained on the Pi from the photos the rig banked and a
human sorted: **J 925, K 722, F 710, and 1232 discards**. The discards are
learnt as a negative class (`_other`, "none of the cats") rather than as a
fourth cat, so a lid can never open for one.

`catbowl train` reported 90.4% accuracy. **That number is wrong and should not
be quoted.** It splits its test set one photo at a time, and the rig captures
every two seconds, so a visit leaves thirty near-identical frames with some in
training and the rest in test - the model was scored on photos it had all but
memorised. Augmentation also runs before the split, so a photo's mirror could
sit in training while the original was tested.

`tools/threshold_report.py` splits by *visit* instead, drops the augmentation,
and reports what a cat walking up actually experiences. Run it on the Pi; it
caches its embeddings, so only the first run costs seven minutes.

    honest accuracy, no floor: 80.6%
    at min_confidence 0.85: 54% of frames accepted, 94.1% of those correct

    per visit          opens for it    opens for the wrong cat    never opens
    K (108 visits)            88.9%                       0.9%         11.1%
    J (166 visits)            64.5%                       3.6%         34.3%
    F (125 visits)            54.4%                       4.8%         43.2%

K works. J is mediocre. **F is the problem** - see below.

Everything lives on the Pi under `data/collected/`, which is gitignored: the
photos exist in exactly one place and no git operation can touch them. The rig
files each new photo it takes into `proposed/<its guess>`, so the review queue
builds itself and every day's photos are a fresh test of the model.

## How a lid decides to open

Opening asks for **one confident sighting** (`policy.open_votes: 1`), not a
consensus. A cat walking up has no history to consult, and every frame it waits
is a frame it spends at a shut bowl. The protection a vote window offers here is
thinner than it looks: consecutive frames of the same cat in the same light are
not independent samples, so a hard frame errs the same way several times over
and the window agrees with itself. A frame the model is genuinely unsure of now
lands in `_other`, which can never win a vote at all.

Everything after the open - intruder, crowd, close - still needs
`votes_required` of the last `vote_window` frames, because by then there is
context to consult. A wrong open costs the wrong cat a mouthful before the
intruder rule shuts the lid; a slow open costs the right cat its meal every
time.

Two guards remain on the open: a cat cannot open the lid while another cat is
better represented in the window, and a sighting that has scrolled out of the
window is not a sighting.

## The two safety questions, answered

**Does it refuse to open when it cannot tell which cat it is?** Yes, by two
independent gates. Any prediction below `recognition.min_confidence` is reported
as `unknown`, and `unknown` can never win a vote (`VoteTracker.decision`). Above
it, a label still has to hold `votes_required` of the last `vote_window` frames
before a lid moves. A cat the model has never seen produces a scatter of
low-confidence guesses, which is a decision of "none".

**Does it know when there is more than one cat?** It does now. This was a real
hole: the classifier is only ever asked "which cat is this crop", and with two
cats at one bowl the honest answer about the crop is still the wrong answer
about the bowl - the other cat is standing right there and eats through the open
lid. The detector is the only part that can see both, so it now counts the cats
it confirms and carries the count on the detection. More than one, and the frame
is reported as `_crowd` instead of a name: it cannot open a lid, and it closes
an open one through the existing intruder path.

Two limits worth knowing. The count comes from `ssdlite`, so it needs
`detector.type: hybrid` (the default) - plain motion detection cannot count. And
between confirmations (`confirm_every_s`, 2 s) the last count stands, so a
second cat arriving is noticed within about two seconds rather than instantly.

## How the one bowl serves three cats

There is one bowl and there are three cats, and there will be one bowl until
there is time to build more. So the bowl feeds all three, on different terms:

| cat | rule |
| --- | --- |
| K | opens whenever she walks up, for as long as she stays |
| J | two minutes of open lid per rolling hour |
| F | two minutes of open lid per rolling hour |

The allowance is a **budget, not one meal an hour**. A cat startled away after
ten seconds keeps the rest of its two minutes and can come back for it. Time is
charged while the lid is open - including the `close_delay_s` it stays up behind
a cat that has already wandered off - and charged as the meal happens, so a cat
that never leaves is still billed. An hour after each mouthful, that mouthful's
worth of allowance comes back.

Running out closes the lid mid-meal (`reason: ration`) and refuses the next
approach with a `denied` event saying how long until it can eat again. A cat
arriving while another is eating still ends the meal - it gets its own turn on
its own allowance, rather than sharing the open lid.

Allowances live in memory. A restart hands every cat a full two minutes again,
which errs towards feeding a cat twice rather than starving one that has eaten
nothing.

## The F problem

F is recognised in a quarter of its frames, and the bowl fails to open for F in
**43% of its visits**. The cause is not what it first looks like.

F's errors do not land on J. They land on `_other` - the model has learnt that F
looks like *junk*, and it learnt that from the labels. F is all black; J has a
white neck and white feet. A tight crop that loses the head and feet leaves a
black shape that could be either, so it gets filed in `discard` - and it is
disproportionately F that gets filed that way, because for J a glimpse of white
settles it. The training set therefore says "black shape with no white = not a
cat", which is exactly what the model now believes.

That is a **labelling feedback loop**, not a shortage of data and not, mainly, a
confusion between two cats. Three things bear on it:

- **Crops were fixed on 2026-09-09** (commit `38ebc5d`): the detector now
  remembers ssdlite's whole-animal box instead of cropping to the tuft of fur
  that moved. Every photo taken before that date has the old, tighter crop, and
  most of the current training set predates it.
- **Judging a visit rather than a frame breaks the loop**, because a visit where
  the feet show in a few frames can name the headless frames beside them. This
  is what `presort` does in batch and what the rig now does live.
- **The plan (2026-09-13) is to wait a week**, let the rig collect well-cropped
  photos, and replace the older F and `discard` photos with them rather than
  re-labelling what is already there.

While waiting, **`proposed/discard` is the tab that needs a human most**. The
model will keep filing F there, and accepting those proposals unchecked would
teach the next model the same bias from its own mistakes.

## What is running right now

- Service `catbowl` on the Pi, in recognition mode (no `--no-model`).
- `min_confidence: 0.85`, `open_votes: 1`, ration 120s/hour for J and F.
- Every capture filed into `data/collected/proposed/<guess>`; visits settled
  live when they end.
- The old cats-only classifier is kept at `~/classifier-catsonly.joblib`, and a
  duplicate photo removed on 2026-09-13 is at `~/duplicates-removed/`.

## Next steps

1. **Collect for a week.** Nothing to do but let it run and check the proposal
   tabs, `proposed/discard` first.
2. **Watch out for `capture.max_images` (5000)**, which counts everything
   unfiled - `unsorted/` plus every `proposed/` folder. Hit it and the rig stops
   banking photos silently.
3. **Replace the old F and discard photos** with the new well-cropped ones. The
   filename timestamps make the pre-2026-09-09 photos easy to pick out.
4. **Retrain**, then re-run `tools/threshold_report.py` and compare the per-visit
   table above. That table, not `train`'s accuracy, is the measure of progress.
5. **Build bowls 2 and 3**, and give J and F their own, at which point the
   rations can go.

## Time, and the four places it is used

A frame is not an independent sample: captures two seconds apart are the same
cat. Four parts of the rig use that, or deliberately do not.

1. **Opening a lid: not at all.** One confident sighting, no history. A cat
   walking up has none to consult.
2. **Closing, intruder, crowd:** `votes_required` of the last `vote_window`
   frames, about a second. By then there is context.
3. **`catbowl presort`:** groups a pile into visits (gap over `--visit-gap`,
   20s) and gives each visit one verdict - the label its confident frames agree
   on, inherited by the unsure ones between them. This cut the unsure pile from
   47% to 23% on the first real run.
4. **The rig's own filing:** the same rule, live. Each photo is filed the
   moment it is taken, so a crash loses nothing; when the visit ends the frames
   are judged together and anything the visit overrules is moved.

Both visit rules refuse themselves where they would do harm: a visit holding
confident frames for two different cats is left per-photo, and a visit nobody
was sure about stays unsure.

Every listing on `/browse` is in time order too - the capture filename carries
the timestamp - so a visit's frames sit together in the grid.

Beyond this the rig carries no memory of who was just here. A visit-level prior
with hysteresis - harder to switch identity mid-meal than to keep it - is the
obvious next refinement once the model is trusted.

## Known problems

- **The camera drops off USB when the servo moves.** Seen on 2026-09-12: the
  lid opened, the kernel logged `USB disconnect` and then `error -71` in a loop,
  and the camera did not come back until it was replugged. `vcgencmd
  get_throttled` was `0x0`, so the Pi itself was not browning out. Hardware
  first - cable routing away from the servo leads, a ferrite on the USB cable, a
  large capacitor across the servo supply. The software should also reopen a
  camera that has dropped; it currently blinds itself until the next restart.
- **Rations reset on restart.** They live in memory, so a reboot hands every cat
  a full allowance. Deliberate - feeding a cat twice beats starving one - but it
  is a loophole if the service restarts often.
- **The status page has no authentication.** `POST /control` moves a physical
  lid and `/sort/*` files photos, for anyone on the wi-fi. A deliberate
  home-LAN trade-off; do not port-forward it. `status_port: null` disables it.
- **`catbowl train` reports a flattering accuracy** (random per-photo split,
  augmentation before the split). Use `tools/threshold_report.py` for a number
  worth acting on. The split inside `training.py` has not been fixed.
- **Second and third bowls.** `bowl2` and `bowl3` are configured but disabled,
  with placeholder cat names. Each needs a camera, a servo and a lid.

## Reference

- **The Pi:** `rjwpi.local` / 192.168.1.241, project at
  `~/Projects/0001_Catbowl`, service `catbowl`, status page on port 8080.
- **Commands** (from the project dir, `catbowl` is not on PATH):

      .venv/bin/python -m catbowl --config config/bowls.yaml doctor
      .venv/bin/python -m catbowl --config config/bowls.yaml train \
          --data data/collected --labels J K F --negative discard
      .venv/bin/python -m catbowl --config config/bowls.yaml presort
      .venv/bin/python tools/threshold_report.py

- **Wiring:** PCA9685 VCC to Pi pin 1 (3.3V), SDA pin 3, SCL pin 5, GND pin 6.
  Servo power from the 5V supply on the 6-pin header, not the screw terminal.
  bowl1's servo is on channel 1, closed 170 degrees, open 85.
