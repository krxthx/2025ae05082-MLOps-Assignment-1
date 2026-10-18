# Live walkthrough recording

The replacement is `../pipeline_walkthrough_narrated.mp4`. It records a real Chromium session: project commands execute through a local console and stream their actual output; MLflow, GitHub Actions, Swagger, Prometheus, and Grafana are live interfaces. Mouse movement, typing, scrolling, and the Swagger prediction submission are captured in the browser video. The EDA plots shown are outputs of the EDA script run during recording.

Narration uses macOS Samantha text-to-speech. It is synthetic, not a recording of the student. The video displays this distinction. `../NARRATION.txt` contains the spoken text for replacement with your own voice if desired. English captions and chapter markers are included. Captions use approximate speech timing.

The scenes were recorded separately and joined. The training wait is shortened; commands, results, and success status are preserved. Diagnostic screenshots are used to review the capture, not to render the video.

## Files

- `console_server.py` and `console.html`: localhost-only command console and source viewer. Only predefined project commands can execute.
- `prepare_audio.py`: synthesizes narration from quoted paragraphs in `../../VIDEO_SCRIPT.md`.
- `record_walkthrough.py`: performs and records the browser interactions.
- `assemble_walkthrough.py`: trims the training wait, adds narration, and exports H.264/AAC MP4 with captions and chapters.
- `timeline.json`: narration and actual exported scene timings.
- `verification.json`: model runs and checks from recording.

## Recording environment

This recording used the existing macOS project environment: Python 3.13, scikit-learn 1.9.1, MLflow 3.17.0, and the existing local Docker/Minikube setup. The project requirements pin MLflow 2.20.1; recording exposed a newer MLflow serialization default, so `src/train.py` now explicitly requests cloudpickle, which is also supported by the pinned version.

Temporary tools are installed under `/private/tmp/assignment-video-tools`: Playwright 1.64.0 and imageio-ffmpeg 0.6.0. The scripts currently use the installed Chromium 1234 executable and temporary assets under `/private/tmp/assignment-recording`. Browser storage files used for local dashboard login stay in that temporary directory and are not included here.

The recording used these host ports to avoid existing services:

| Service | Port |
|---|---|
| Isolated Docker API | 8001 |
| Kubernetes port-forward | 8002 |
| Compose API | 8010 |
| MLflow | 5001 |
| Prometheus | 9091 |
| Grafana | 3001 |
| Recording console | 8766 |

The Docker API and Kubernetes API return the same sample prediction. Monitoring traffic goes to the Compose API. GitHub Actions footage shows a previously completed real workflow run; it does not represent a new push or deployment during recording.

To capture again, prepare those services and the browser login state, run `prepare_audio.py`, start `console_server.py`, then run `record_walkthrough.py` with `PLAYWRIGHT_BROWSERS_PATH=/private/tmp/assignment-pw-browsers`. Individual scene numbers can be passed to retake only those scenes. Before retaking the Docker scene, remove only the `heart-video` container created for this recording, or change that scene's container name. Finally run `assemble_walkthrough.py`.

## October 18 refresh

Repository, commit, CI, and MLflow views were recaptured after the author corrected the GitHub history. The CI footage uses successful run #7 at commit `5c2a38f`. MLflow is recorded with its assistant panel closed through the normal interface. The other five pipeline sections retain the verified October 17 footage and narration.

The final export remains `../pipeline_walkthrough_narrated.mp4`; no silent alternative is included. `refresh_walkthrough.py` rebuilds the export from retained segments and the four new takes, preserving the existing narration, captions, and chapter timing. Its inputs are the temporary previous export, original timeline, scene audio, retained segments, and new browser recordings under `/private/tmp/assignment-recording`. Browser navigation waits are compressed where needed to preserve narration timing.
