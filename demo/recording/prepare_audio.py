import sys
from pathlib import Path
import re
import subprocess
sys.path.insert(0,'/private/tmp/assignment-video-tools')
import imageio_ffmpeg
root=Path(__file__).resolve().parents[2]
work=Path('/private/tmp/assignment-recording')
work.mkdir(exist_ok=True)
s=(root/'VIDEO_SCRIPT.md').read_text()
scenes=[]
for section in re.split(r'^## ',s,flags=re.M)[1:]:
 speech=' '.join(line[2:] for line in section.splitlines() if line.startswith('> '))
 if not speech: continue
 index=len(scenes)+1
 text=work/f'{index:02}.txt';text.write_text(speech)
 subprocess.run(['say','-v','Samantha','-r','151','-f',str(text),'-o',str(work/f'{index:02}.aiff')],check=True)
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(work/f'{index:02}.aiff'),str(work/f'{index:02}.wav')],check=True)
 import wave
 with wave.open(str(work/f'{index:02}.wav')) as f: duration=f.getnframes()/f.getframerate()
 scenes.append({'index':index,'title':section.splitlines()[0],'speech':speech,'duration':duration})
 print(index,round(duration,1),flush=True)
import json
(work/'scenes.json').write_text(json.dumps(scenes,indent=2))
