# Clip Sleuth Linux download

This is the experimental Linux x86-64 desktop bundle. It includes Python,
Clip Sleuth, Tk and the Python dependencies. It does **not** include FFmpeg.
Install `ffmpeg` and `ffprobe` with your Linux distribution's package manager;
both must be on `PATH`. A graphical desktop session is required for the GUI.

Extract the archive (keep the folder together), then run:

```sh
./clip-sleuth/clip-sleuth
```

Choose **Index a source library**, then **Match a clip folder**. Use **Open
verified report** only after a report finishes. The archive also offers a CLI
worker for diagnostics, for example `./clip-sleuth/clip-sleuth --gui-worker
--help`. A download is not an installer, and no desktop menu entry is added.

Built on Linux x86-64 with glibc 2.35; exercised on Ubuntu 22.04 and one newer
Kubuntu desktop. Other distributions and CPU architectures are not qualified.
The app may use substantial memory and decoding time on long videos. See the
repository README for supported media, confidence limitations and local data
privacy. Indexes, reports and evidence remain on your computer. The bundled
application is MIT-licensed; FFmpeg is provided separately by your system.
