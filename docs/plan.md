# The state of the catbowl

Where the rig is, what is wrong with it, and what happens next.
Updated 2026-09-14 (retrained that night, on the re-sorted discard pile). Keep it updated: it is the only
place most of this is written down.

## Where it is now

The hardware works. One bowl is live (`bowl1`): a cardboard disc lying on the
bowl, turned aside by a single servo on PCA9685 channel 1, now glued to the lid
rather than taped. The camera is a Camera Module 3 Wide (imx708, 120 degrees) on the ribbon
cable, since 2026-09-22; the Logitech Brio 100 webcam before it is removed. The status page
serves the live view, the sorting queue and the browser at
`http://rjwpi.local:8080/`.

A classifier now exists, trained from the photos the rig banked and a human
sorted: **J 1206, K 815, F 958**, and two folders learnt together as "not a
cat": **`discard` 378** (no cat) and **`M` 276** (several cats). A third,
**`unclear` 890** (a cat, but F or J cannot be told), is kept and not trained.
Retrained 2026-09-14 on the desktop rather than the Pi - forty seconds instead
of seven minutes, and the Pi only receives the finished
`models/classifier.joblib`. The negatives are learnt as a negative class (`_other`, "none of the cats") rather
than as a fourth cat, so a lid can never open for one.

`catbowl train` now reports **85.2%**, and since 2026-09-14 that number is
honest: it holds out whole visits, keeps each mirrored copy with its original,
and then refits the saved model on every photo. Before that date it split its
test set one photo at a time, and the rig captures every two seconds, so a visit leaves thirty near-identical frames with some in
training and the rest in test - the model was scored on photos it had all but
memorised. Augmentation also runs before the split, so a photo's mirror could
sit in training while the original was tested.

`tools/threshold_report.py` splits by *visit* instead, drops the augmentation,
and reports what a cat walking up actually experiences. Run it on the desktop
mirror; it caches its embeddings, so only new photos are embedded.

    honest accuracy, no floor: 81.4%
    at min_confidence 0.85: 51.5% of frames accepted, 95.6% of those correct

    per visit          opens for it    opens for the wrong cat    never opens
    K (121 visits)            89.3%                       1.7%         10.7%
    J (195 visits)            62.1%                       7.2%         35.4%
    F (156 visits)            69.9%                       3.8%         26.9%
    junk (165 visits)          7.9%   <- would have opened a lid

Visits are grouped by time alone since 2026-09-14, whatever folder their
frames were filed in, so these are not directly comparable with the rows
below. The like-for-like comparison of the re-sort, same visits and same
folds for each (`tools/resort_ablation.py`, per visit at 0.85):

    what "not a cat" was trained on       F      J      K   junk  crowd opens
    discard + M + unclear (before)      64.8   63.4   89.2   4.3     33.3
    discard only (M, unclear left out)  73.2   68.9   89.2   2.9     58.3
    discard + M, unclear left out  <--  74.6   62.8   89.2   5.1     25.0
    discard, and M as a crowd class     69.0   59.6   87.5   1.4     19.4

The last would also close an open lid on about half of crowds, since it is
the only one that can name a crowd; it cost F five points and was not chosen.
Leaving M out of training entirely is unsafe: the detector misses most crowds,
and the classifier is the real defence against them.

The history, per visit (K / J / F):

    2026-09-13 am   3589 photos   88.9 / 64.5 / 54.4   raw 80.6%
    2026-09-13 pm   4153 photos   90.4 / 57.9 / 59.7   raw 75.4%
    2026-09-14 am   4434 photos   88.4 / 59.2 / 61.0   raw 76.9%
    2026-09-14 pm   3633 trained  89.3 / 62.1 / 69.9   raw 81.4%  (re-sorted;
                    visits now grouped by time, so a slightly different count)

Between the first two runs: **F improved by five points and J lost seven.** The raw
number fell too, which is what a harder and more varied test set looks like
rather than a worse model - the added photos come mostly from the new wide
crops, so the model is now being asked a fairer question. The wrong-cat rate
fell everywhere.

K works. F and J are now held back by each other rather than by the junk pile:
since the re-sort, a missed F frame is far more often called J than "not a
cat" - see The F problem.

Everything lives on the Pi under `data/collected/`, which is gitignored, so
no git operation can touch the photos. The Pi's copy is the master; the desktop
holds an rsync mirror for training. The rig
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
| K | opens whenever she walks up, and stays open until she leaves (`uncapped`) |
| J | one minute of open lid per rolling hour |
| F | one minute of open lid per rolling hour |

The allowance is a **budget, not one meal an hour**. A cat startled away after
ten seconds keeps the rest of its minute and can come back for it. Time is
charged while the lid is open - including the `close_delay_s` it stays up behind
a cat that has already wandered off - and charged as the meal happens, so a cat
that never leaves is still billed. An hour after each mouthful, that mouthful's
worth of allowance comes back.

Running out closes the lid mid-meal (`reason: ration`) and refuses the next
approach with a `denied` event saying how long until it can eat again. A cat
arriving while another is eating still ends the meal - it gets its own turn on
its own allowance, rather than sharing the open lid.

Allowances live in memory. A restart hands every cat a full minute again,
which errs towards feeding a cat twice rather than starving one that has eaten
nothing.

## The F problem

F used to open the bowl in barely half its visits. The cause was not a
shortage of photos and not, mainly, confusion with J: F's misses landed on
`_other`. The model had learnt that F looks like *junk*, and it learnt it from
the labels.

F is all black; J has a white neck and white feet. A crop that loses the head
and feet leaves a black shape that could be either, and those were filed in
`discard` - disproportionately F's, because for J a glimpse of white settles
it. So the training set said "black shape with no white = not a cat".

**Fixed on 2026-09-14 by splitting the discard pile** into what it actually
held: 378 photos with no cat (`discard`), 276 with several (`M`) and 890 of
one cat that could not be named (`unclear`). `unclear` is no longer trained at
all, and F went from 64.8% to 74.6% of visits on the same folds (about 83% on
well-cropped visits). F's misses now land on J rather than on junk - the
confusion matrix moved from 36-to-junk to 36-to-J out of 171 - which is the
honest version of the problem: two cats that genuinely look alike from behind.

Two other things bore on it and still do:

- **Crops were fixed in commit `38ebc5d`, dated 2026-09-09 - but that is the
  commit date, not the deploy date.** The Pi's reflog shows its checkout sat on
  a 2026-09-03 commit until it was pulled forward at **2026-09-12 21:00**, and
  the photos agree: the fraction of tiny crops (under 40k px) falls from ~13% to
  under 1% overnight. Verify such things the same way rather than trusting a
  commit date: `git reflog --date=iso` on the Pi, against crop areas by hour.
- **Old crops still help, for now.** Tested on well-cropped visits, a model
  trained on only the new crops did worse than one trained on everything (F
  72% vs 90%, J 38% vs 71%); putting K's old photos back recovered K but not F
  or J. There are too few new crops to stand alone. Re-run the comparison
  when the new crops roughly equal the old.

Keep `proposed/discard` and `proposed/unsure` honest when reviewing: a
headless black cat belongs in `unclear`, not in `discard`.

## What is running right now

- Service `catbowl` on the Pi, in recognition mode.
- Classifier trained 2026-09-22 on `J K F` with `--negative discard M`
  (4129 photos); `unclear` left out. Honest score 85.2% (held-out visits);
  per visit at 0.85: K 90%, J 63%, F 72%, junk 9%. J opening under F's name
  down from 7% to 5% of J's visits.
- `min_confidence: 0.85`, `open_votes: 1`.
- Rations: J and F 60s of open lid per rolling hour (halved from 120s on
  2026-09-13). K has none.
- `max_open_s: 30` for J and F. K is `uncapped` (2026-09-14): her lid stays up
  until she leaves or another cat arrives.
- Servo: channel 1, closed 175 degrees, open 65.
- Camera: `csi:0` at 1280x720 through picamera2. The venv is built with
  `--system-site-packages`, because picamera2 only comes from apt. The model
  was trained on webcam photos only. Continuous autofocus is on.
- Cat detector: YOLO11n at 640px (`detector.model: yolo`, the ONNX file in
  `models/`, not in git), replacing ssdlite on 2026-09-22.
- Sort buckets: `J K F M unclear` plus `discard`. Every capture is filed into
  `data/collected/proposed/<guess>`; visits are settled live when they end.
- Pi health (2026-09-14, `scripts/install_pi_health.sh`): the journal is kept
  on disk, Wi-Fi power saving is off, and `netmon` logs reachability, power,
  temperature and memory to `~/netmon.log` every 15s.
- Backups on the Pi: `~/classifier-catsonly.joblib` (first model),
  `~/classifier-2026-09-06.joblib`, `~/classifier-2026-09-13.joblib` and
  `~/classifier-2026-09-14am.joblib`, `~/classifier-2026-09-22.joblib` (the ones each retrain replaced), and a
  duplicate photo removed on 2026-09-13 in `~/duplicates-removed/`.

## Next steps

0. **Retrain on Pi-camera photos.** The new camera sees wider, with other
   colours, than the webcam every training photo came from, so recognition may
   be worse until a few days of its photos are sorted and trained on.
1. **Watch F and J's meals.** 7% of J's visits now open under F's name, which
   spends F's minute on J. The events log (`logs/events-*.jsonl`) shows who
   each lid opened for; if F keeps running out of ration, this is why.
2. **Keep sorting**, `proposed/unsure` and `proposed/discard` first, into the
   five buckets. `M` is the thinnest class that matters (276 photos, 36
   visits); more crowds make the refusal numbers trustworthy.
3. **If the Pi drops off the network again**, note the time and read
   `~/netmon.log` and `journalctl -b -1` around it. `net=FAIL` with the Pi still
   up is a network stall; a gap in the log is a crash or a hang.
4. **Watch `capture.max_images` (5000)**, which counts `unsorted/` plus every
   `proposed/` folder. Hit it and the rig stops banking photos silently.
5. **Retrain** on the desktop (commands under Reference) and compare the
   per-visit table. Re-run the old-vs-new crop comparison once the new crops
   roughly equal the old, and prune the old ones only if it says so.
6. **Consider a crowd class.** Training `M` as its own class (not "not a cat")
   was measured: it refuses more crowds (19% open vs 25%) and can close an open
   lid on about half of them - the only defence if a cat joins K at her
   uncapped lid - at a cost of five points of F. Not chosen on 2026-09-14,
   because F needed the help more.
7. **Build bowls 2 and 3**, and give J and F their own, at which point the
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

- **The camera dropped off USB when the servo moved** (webcam, now replaced). Seen on 2026-09-12: the
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
- **The Pi went off Wi-Fi repeatedly** (through 2026-09-14), sometimes
  needing a reboot. The logs from those crashes were lost - Raspberry Pi OS
  keeps the journal in memory - so the cause is not proven. What was seen:
  strong signal (-55 dBm), never throttled, 50 degrees C, plenty of memory, and
  no Wi-Fi disconnect in the Pi's own log while `rjwpi.local` stopped
  resolving. That is the signature of the brcmfmac chip's power saving, which
  was on; it is now off. The journal is now kept and `netmon` is running, so
  a recurrence will leave a record.
- **Unclear frames still happen at the bowl.** `unclear/` is left out of
  training, not out of the world. Shown one, the model hedges between F and J;
  mostly that stays under 0.85 and does nothing (no open, no close), and the
  next frame showing a face or white fur decides. About a third of unclear
  visits do produce a confident call, and that is where the wrong-cat opens
  come from.
- **Second and third bowls.** `bowl2` and `bowl3` are configured but disabled,
  with placeholder cat names. Each needs a camera, a servo and a lid.

## Reference

- **The Pi:** `rjwpi.local` / 192.168.1.241, project at
  `~/Projects/0001_Catbowl`, service `catbowl`, status page on port 8080.
- **Pi health:** `scripts/install_pi_health.sh` reinstalls the journal,
  Wi-Fi and `netmon` fixes (files in `pi/` and `systemd/netmon.service`).
- **Commands** (from the project dir, `catbowl` is not on PATH):

      .venv/bin/python -m catbowl --config config/bowls.yaml doctor
      # Training runs on the desktop; only the .joblib goes to the Pi.
      rsync -a rjweldon@rjwpi.local:Projects/0001_Catbowl/data/collected/ data/collected/
      .venv/bin/python -m catbowl --config config/bowls.yaml train \
          --data data/collected --labels J K F --negative discard M
      scp models/classifier.joblib rjweldon@rjwpi.local:Projects/0001_Catbowl/models/
      .venv/bin/python -m catbowl --config config/bowls.yaml presort
      .venv/bin/python tools/threshold_report.py

- **Wiring:** PCA9685 VCC to Pi pin 1 (3.3V), SDA pin 3, SCL pin 5, GND pin 6.
  Servo power from the 5V supply on the 6-pin header, not the screw terminal.
  bowl1's servo is on channel 1, closed 175 degrees, open 65.

## Log

**2026-09-13.** Multi-cat frames refused (`_crowd` from the detector's count).
Live self-sorting into `proposed/<guess>`, settled per visit. Opening on one
confident frame (`open_votes: 1`). One bowl for all three cats with rolling
rations, 120s/hour for J and F. Discard trained as a negative class.
`threshold_report.py` written: the first honest score (80.6%, not 90.4%).
Retrained that evening on 4153 photos; rations halved to 60s/hour. Found that
the crop fix reached the Pi on the 12th, not the 9th.

**2026-09-14.** Retrained on 4434 photos. K made `uncapped`. A visit settled
as `_other` was being filed into `proposed/_other`; now `proposed/discard`.
Measured old vs new crops (keep the old, for now). Added `M` and `unclear`
buckets; the discard pile was re-sorted by hand into 378 / 276 / 890; four
ways of training them compared, and "discard + M as not-a-cat, unclear left
out" chosen and deployed (F 65% -> 75% of visits). `train` now splits by
visit, keeps mirrors with their originals and refits on every photo. The Pi's
Wi-Fi drops investigated: journal made persistent, `netmon` installed, power
saving turned off.

**2026-09-22.** Webcam replaced by a Camera Module 3 Wide on the ribbon;
`bowl1` now uses `csi:0` at 1280x720, and the Pi's venv was switched to see
system packages so it can import picamera2.
Continuous autofocus turned on. F sitting at the bowl was not being seen as a
cat: on 20 minutes of live frames ssdlite found the cat in 47 of 94 frames
(mostly F, head-down), YOLO11n at 320px 69, at 640px 93, with no false alarms
in 261 empty frames for any of them. Switched to YOLO11n 640 (0.6s a check on
the Pi, against ssdlite's 0.7s). Still open: the gate only asks the detector
when something moves, so a cat sitting perfectly still is not looked for.
Retrained on 4129 photos (J 1402, K 836, F 1093, M 360, discard 438; unclear
965 left out): F 72%, J 63%, K 90% of visits, junk 9%.

**2026-09-24.** Retrained on 4872 photos with `--negative discard` only (M left
out of training): F 73%, J 64%, K 90% of visits; junk 3%; J opens as another
cat 9% (was 5%). Previous model backed up as `~/classifier-2026-09-24.joblib`.

Switched to a model trained on Pi-camera photos only (since 2026-09-22 09:48;
J K F + discard, no M, no unclear) at `min_confidence: 0.75`: per visit F 75%
(3% wrong cat), J 64% (5%), K 89%, empty 9% (3 of 34). Scored on the same
Pi-camera visits, the all-photos model at 0.85 gave F 64%, J 48% (9% wrong).
Nightly retrain installed: `catbowl-train.timer` runs `tools/nightly_train.py`
at 00:00, retrains only if the sorted Pi-camera set changed, backs up to
`models/backups/`, logs scores to `models/history.json` (shown on the status
page) and restarts catbowl. About 6 minutes on the Pi.
